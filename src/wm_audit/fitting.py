"""Training-only physical probes and small, full-history latent editors.

``PhysicalReadout`` consumes the *emitted* state: projected LeWM CLS (192,)
or native DINO patches (196,384). Its fixed DINO transform is channel-first
adaptive average pooling of the 14x14 patch grid to 2x2 (1536 features).
LeWM is not pooled. No encoder, random projection, or temporal averaging is
substituted for that representation. Predictions are in physical units.

``fit_editor(name, histories, requests, targets, ...) -> (editor, log)`` fits
only the supplied training pairs. The caller must enforce disjoint root/episode
splits. Evaluation scorers are separate PhysicalReadout instances; an inverse
editor constructs and fits its OWN probe on factual frame 0 / anchor labels.
Requests are CF-minus-factual anchor deltas, including differences of sin/cos,
not an angle or a desired endpoint offset. Neither labels nor native targets
are inputs at inference.

Editors keep a full [rank,3,*latent_shape] basis, NOT a flattened dense ridge
map. An uncentered, seeded CPU randomized SVD fits that basis to at most 256
paired differences by default (configurable via basis_samples). Rank is a
capacity limit, not a claim about the full data rank. Editor-only conditioning
uses training PCA of compact, pooled factual histories. Ridge/MLP inputs are
[r, r outer context], where r uses fixed [32,32,32,32,1,1] scales. All editors
are exactly identity for zero requests, even after learning nonlinear biases.

Large source tensors/NumPy arrays can stay on CPU. Only compact features and
coefficients are retained; full histories enter the fitting device in batches.
Rollout supervision is against supplied native-model CF futures, or native
rollouts of target histories computed on demand, never simulator future latents
implicitly. Actions are native [N,K,2+H,10] (also [N,2+H,10] for K=1).
The model must already be on the fitting device. It is frozen/eval during the
fit, with its original flags restored afterward; history gradients stay live.

The JSON-compatible log records effective rank, sampled basis size, initial/
final training history MSE, and per-update losses (not validation estimates).
Readout state_dicts and save_editor checkpoints contain tensor dictionaries,
loadable with torch.load(..., weights_only=True); no module is pickled.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


__all__ = ["PhysicalReadout", "make_editor", "fit_editor", "save_editor",
           "load_editor", "matched_random"]

_STATE_SCALE = (512.0, 512.0, 512.0, 512.0, 1.0, 1.0)
_REQUEST_SCALE = (32.0, 32.0, 32.0, 32.0, 1.0, 1.0)
_NAMES = ("identity", "inverse", "ridge", "mlp", "rollout")


def _integer(value, label, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}.")
    return int(value)


def _real(value, label, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}.")
    return float(value)


def _shape(latent_shape, grid):
    try:
        shape = tuple(_integer(d, "latent dimension") for d in latent_shape)
    except TypeError as exc:
        raise ValueError("latent_shape must be a vector or square patch-grid shape.") from exc
    if len(shape) not in (1, 2):
        raise ValueError("latent_shape must be (features,) or (square_patches, channels).")
    if len(shape) == 2 and (math.isqrt(shape[0]) ** 2 != shape[0] or grid > math.isqrt(shape[0])):
        raise ValueError("Patch count must be square and grid must not exceed its side.")
    return shape


def _tensor(value, label):
    try:
        value = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise TypeError(f"{label} must be a floating tensor or array.") from exc
    if not value.is_floating_point():
        raise TypeError(f"{label} must be floating point.")
    return value


def _finite(value, label):
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} must be finite.")
    return value


def _states(value, tail, label, batched=False):
    value = _tensor(value, label)
    if value.ndim < len(tail) or tuple(value.shape[-len(tail):]) != tail:
        raise ValueError(f"{label} must end in shape {tail}.")
    prefix = tuple(value.shape[:-len(tail)])
    if batched and (len(prefix) != 1 or prefix[0] < 1):
        raise ValueError(f"{label} must have shape [N,{','.join(map(str, tail))}], N > 0.")
    return value, prefix


def _batch(value, index, device="cpu", dtype=torch.float32):
    # Detach TRAINING inputs, not editor.forward or the model rollout graph.
    part = value[index] if isinstance(index, slice) else value.index_select(0, index.to(value.device))
    return _finite(part.detach().to(device=device, dtype=dtype), "training batch")


def _feature_dim(shape, grid):
    return shape[0] if len(shape) == 1 else shape[1] * grid * grid


def _pool(value, shape, grid):
    if len(shape) == 1:
        return value
    prefix, channels, side = value.shape[:-2], shape[1], math.isqrt(shape[0])
    if value.numel() == 0:
        return value.new_empty(*prefix, channels * grid * grid)
    image = value.reshape(-1, *shape).transpose(1, 2).reshape(-1, channels, side, side)
    return F.adaptive_avg_pool2d(image, (grid, grid)).reshape(*prefix, channels * grid * grid)


def _standardize(x):
    mean, std = x.mean(0), x.std(0, correction=0)
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    return mean, std


def _ridge(x, y, strength, intercept):
    """Sum-squared-error + strength*||W||^2; never penalize the intercept."""
    x, y = x.double(), y.double()
    xm = x.mean(0) if intercept else x.new_zeros(x.shape[1])
    ym = y.mean(0) if intercept else y.new_zeros(y.shape[1])
    x, y = x - xm, y - ym
    if strength == 0:
        weight = torch.linalg.lstsq(x, y, driver="gelsd").solution
    elif x.shape[0] < x.shape[1]:
        gram = x @ x.T
        gram.diagonal().add_(strength)
        weight = x.T @ torch.linalg.solve(gram, y)
    else:
        gram = x.T @ x
        gram.diagonal().add_(strength)
        weight = torch.linalg.solve(gram, x.T @ y)
    return weight, ym - xm @ weight


class PhysicalReadout(nn.Module):
    """Per-state standardized linear ridge with outputs in physical units.

    fit: [N,*latent_shape], [N,6] -> self. predict/forward accepts any prefix,
    including a single state or [N,K,H,*latent_shape], and returns [*,6].
    State dicts can be restored into a fresh instance with the same shape/grid.
    Fit uses CPU float64 solves; prediction uses the module's device/dtype and
    preserves gradients. Constant training features have standard deviation 1.
    """

    def __init__(self, latent_shape, grid=2, ridge=1e-2):
        super().__init__()
        self.grid = _integer(grid, "grid")
        self.latent_shape = _shape(latent_shape, self.grid)
        self.feature_dim = _feature_dim(self.latent_shape, self.grid)
        self.register_buffer("ridge_strength", torch.tensor(_real(ridge, "ridge"), dtype=torch.float64))
        self.register_buffer("input_mean", torch.zeros(self.feature_dim))
        self.register_buffer("input_std", torch.ones(self.feature_dim))
        self.register_buffer("weight", torch.zeros(self.feature_dim, 6))
        self.register_buffer("bias", torch.zeros(6))
        self.register_buffer("output_scale", torch.tensor(_STATE_SCALE))
        self.register_buffer("fitted", torch.tensor(False))

    @property
    def ridge(self):
        return self.ridge_strength.item()

    def features(self, z):
        """The fixed, unstandardized per-state transform; no fitting or RNG."""
        z, _ = _states(z, self.latent_shape, "z")
        z = _finite(z.to(device=self.weight.device, dtype=self.weight.dtype), "z")
        return _pool(z, self.latent_shape, self.grid)

    @torch.no_grad()
    def fit(self, z, y):
        z, (n,) = _states(z, self.latent_shape, "z", batched=True)
        y, prefix = _states(y, (6,), "y", batched=True)
        if prefix != (n,):
            raise ValueError("z and y must have matching N.")
        x = torch.empty(n, self.feature_dim, dtype=torch.float64)
        for start in range(0, n, 64):
            index = slice(start, start + 64)
            x[index] = _pool(_batch(z, index, dtype=torch.float64), self.latent_shape, self.grid)
        return self.fit_features(x, y)

    @torch.no_grad()
    def fit_features(self, x, y):
        """Fit pre-pooled features, avoiding storage of all DINO patch tensors.

        Only features produced by this instance's fixed ``features`` transform
        may be supplied. This is equivalent to fit(z, y), not a new surface.
        """
        x = _finite(torch.as_tensor(x, dtype=torch.float64, device="cpu"), "features")
        y = torch.as_tensor(y)
        if x.ndim != 2 or x.shape[1] != self.feature_dim or y.shape != (len(x), 6) or len(x) < 1:
            raise ValueError("Expected nonempty [N,feature_dim] features and [N,6] labels.")
        mean, std = _standardize(x)
        labels = _batch(y, slice(None), dtype=torch.float64) / self.output_scale.cpu().double()
        weight, bias = _ridge((x - mean) / std, labels, self.ridge, intercept=True)
        for name, value in (("input_mean", mean), ("input_std", std), ("weight", weight), ("bias", bias)):
            getattr(self, name).copy_(value)
        self.fitted.fill_(True)
        return self

    def predict(self, z):
        if not self.fitted.item():
            raise RuntimeError("PhysicalReadout must be fitted before prediction.")
        z, prefix = _states(z, self.latent_shape, "z")
        rows = z.reshape(-1, *self.latent_shape)
        predictions = []
        for start in range(0, rows.shape[0], 64):
            x = (self.features(rows[start:start + 64]) - self.input_mean) / self.input_std
            predictions.append((x @ self.weight + self.bias) * self.output_scale)
        if not predictions:
            return self.weight.new_empty(*prefix, 6)
        return torch.cat(predictions).reshape(*prefix, 6)

    def forward(self, z):
        return self.predict(z)


class _HistoryEditor(nn.Module):
    def __init__(self, name, latent_shape, rank, seed, grid, context_dim, hidden_dim, ridge):
        super().__init__()
        if name not in _NAMES:
            raise ValueError(f"name must be one of {_NAMES}.")
        self.name, self.rank, self.seed = name, _integer(rank, "rank"), _integer(seed, "seed", 0)
        if self.seed >= 2 ** 63:
            raise ValueError("seed must fit a nonnegative int64.")
        self.grid = _integer(grid, "grid")
        self.latent_shape = _shape(latent_shape, self.grid)
        width = 3 * _feature_dim(self.latent_shape, self.grid)
        self.context_dim = min(_integer(context_dim, "context_dim"), width)
        self.hidden_dim, self.ridge = _integer(hidden_dim, "hidden_dim"), _real(ridge, "ridge")
        self.register_buffer("basis", torch.zeros(self.rank, 3, *self.latent_shape))
        self.register_buffer("effective_rank", torch.tensor(0))
        self.register_buffer("context_mean", torch.zeros(width))
        self.register_buffer("context_std", torch.ones(width))
        self.register_buffer("context_projection", torch.zeros(width, self.context_dim))
        self.register_buffer("context_scale", torch.ones(self.context_dim))
        self.register_buffer("coefficient_scale", torch.ones(self.rank))
        self.register_buffer("request_scale", torch.tensor(_REQUEST_SCALE))
        self.register_buffer("fitted", torch.tensor(False))
        inputs = 6 if name in ("identity", "inverse") else 6 * (1 + self.context_dim)
        self.linear = nn.Linear(inputs, self.rank, bias=False)
        nn.init.zeros_(self.linear.weight)
        self.mlp = None
        if name in ("mlp", "rollout"):
            self.mlp = nn.Sequential(nn.Linear(inputs, self.hidden_dim), nn.SiLU(),
                                     nn.Linear(self.hidden_dim, self.rank))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            self.linear.requires_grad_(False)
        self.probe = PhysicalReadout(self.latent_shape, grid, ridge) if name == "inverse" else None

    def _inputs(self, history, request):
        history, (n,) = _states(history, (3, *self.latent_shape), "history", batched=True)
        request, prefix = _states(request, (6,), "request", batched=True)
        if prefix != (n,):
            raise ValueError("history and request must have matching N.")
        history = _finite(history.to(device=self.basis.device), "history")
        request = _finite(request.to(device=self.basis.device, dtype=self.basis.dtype), "request")
        return history, request

    def _context(self, features):
        return ((features - self.context_mean) / self.context_std) @ self.context_projection / self.context_scale

    def _design(self, context, request):
        r = request / self.request_scale
        if self.name in ("identity", "inverse"):
            return r
        return torch.cat((r, (r.unsqueeze(2) * context.unsqueeze(1)).flatten(1)), dim=1)

    def _coefficients(self, design):
        if self.name == "identity":
            return design.new_zeros(design.shape[0], self.rank)
        coefficient = self.linear(design)
        if self.mlp is not None:
            coefficient = coefficient + self.mlp(design) - self.mlp(torch.zeros_like(design))
        coefficient = coefficient * self.coefficient_scale
        return torch.where(design.ne(0).any(1, keepdim=True), coefficient, torch.zeros_like(coefficient))

    def _from_history(self, history, request):
        context = None
        if self.name not in ("identity", "inverse"):
            features = _pool(history.to(dtype=self.basis.dtype), self.latent_shape, self.grid).flatten(1)
            context = self._context(features)
        return self._coefficients(self._design(context, request))

    def coefficients(self, history, request):
        """Actual full-history basis coefficients, useful for matched controls."""
        history, request = self._inputs(history, request)
        return self._from_history(history, request)

    def _apply_edit(self, history, coefficient, request):
        delta = (coefficient @ self.basis.flatten(1)).reshape_as(history).to(dtype=history.dtype)
        active = request.ne(0).any(1).reshape(-1, *([1] * (history.ndim - 1)))
        # Preserve the input dtype and bitwise sham values, not just small loss.
        return torch.where(active, history + delta, history)

    def forward(self, history, request):
        history, request = self._inputs(history, request)
        if self.name == "identity":
            return history
        return self._apply_edit(history, self._from_history(history, request), request)


def make_editor(name, latent_shape, rank=8, seed=0, *, grid=2, context_dim=16,
                hidden_dim=64, ridge=1e-2) -> nn.Module:
    """Construct an initially identity editor without changing global RNG state."""
    seed = _integer(seed, "seed", 0)
    if seed >= 2 ** 63:
        raise ValueError("seed must fit a nonnegative int64.")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return _HistoryEditor(name, latent_shape, rank, seed, grid, context_dim, hidden_dim, ridge)


def _pca_rows(x, rank, seed):
    """Uncentered, CPU-only approximate right singular vectors, with zero padding elsewhere."""
    q = min(rank + 4, *x.shape)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        _, singular, vectors = torch.pca_lowrank(x, q=q, center=False, niter=2)
    tolerance = singular[0] * torch.finfo(x.dtype).eps * max(x.shape[0], q)
    effective = min(rank, int((singular > tolerance).sum().item()))
    rows = vectors[:, :effective].T.contiguous()
    if effective:
        signs = rows.gather(1, rows.abs().argmax(1, keepdim=True)).sign()
        rows.mul_(signs)
    return rows, effective


@torch.no_grad()
def _fit_basis(editor, histories, targets, count, batch_size):
    n, width = histories.shape[0], editor.basis[0].numel()
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(editor.seed))[:count]
    differences = torch.empty(len(indices), width)
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start + batch_size]
        differences[start:start + len(chosen)] = (_batch(targets, chosen) - _batch(histories, chosen)).flatten(1)
    rows, effective = _pca_rows(differences, editor.rank, editor.seed)
    editor.basis.flatten(1)[:effective].copy_(rows)
    editor.effective_rank.fill_(effective)
    return len(indices)


@torch.no_grad()
def _fit_context(editor, histories, batch_size):
    features = torch.empty(histories.shape[0], editor.context_mean.numel())
    for start in range(0, histories.shape[0], batch_size):
        index = slice(start, start + batch_size)
        features[index] = _pool(_batch(histories, index), editor.latent_shape, editor.grid).flatten(1)
    mean, std = _standardize(features)
    editor.context_mean.copy_(mean)
    editor.context_std.copy_(std)
    standardized = (features - mean) / std
    rows, effective = _pca_rows(standardized, editor.context_dim, (editor.seed + 1) % (2 ** 63))
    editor.context_projection[:, :effective].copy_(rows.T)
    projected = standardized @ editor.context_projection
    editor.context_scale.copy_(_standardize(projected)[1])
    return projected / editor.context_scale


@torch.no_grad()
def _project_targets(editor, histories, targets, batch_size):
    n, width = histories.shape[0], editor.basis[0].numel()
    coefficients, residuals = torch.empty(n, editor.rank), torch.empty(n, dtype=torch.float64)
    for start in range(0, n, batch_size):
        index = slice(start, start + batch_size)
        difference = (_batch(targets, index) - _batch(histories, index)).flatten(1)
        c = difference @ editor.basis.flatten(1).T
        coefficients[index] = c
        energy = difference.square().sum(1, dtype=torch.float64)
        residuals[index] = (energy - c.double().square().sum(1)).clamp_min(0) / width
    return coefficients, residuals


@torch.no_grad()
def _history_mse(editor, design, coefficients, residuals, batch_size):
    total, width = 0.0, editor.basis[0].numel()
    for start in range(0, design.shape[0], batch_size):
        index = slice(start, start + batch_size)
        predicted = editor._coefficients(_batch(design, index, editor.basis.device)).cpu().double()
        total += ((predicted - coefficients[index].double()).square().sum(1) / width + residuals[index]).sum().item()
    return total / design.shape[0]


def _rollout_data(model, actions, future_targets, n, shape, device, batch_size):
    if not isinstance(model, nn.Module) or not callable(getattr(model, "rollout", None)):
        raise ValueError("rollout fitting requires an nn.Module with model.rollout(history, actions).")
    if any(t.device != device for t in (*model.parameters(), *model.buffers())):
        raise ValueError("model must already be on the fitting device.")
    actions = _tensor(actions, "actions")
    if actions.ndim == 3:
        actions = actions.unsqueeze(1)
    if (actions.ndim != 4 or actions.shape[0] != n or actions.shape[1] < 1
            or actions.shape[2] < 3 or actions.shape[3] != 10):
        raise ValueError("actions must be [N,K,2+H,10] or [N,2+H,10], K,H >= 1.")
    _finite(actions, "actions")
    if (actions.abs() > 1).any():
        raise ValueError("actions must be native controls in [-1,1], not z-scored.")
    if future_targets is not None:
        future_targets = _tensor(future_targets, "future_targets")
        if future_targets.ndim == 2 + len(shape) and actions.shape[1] == 1:
            future_targets = future_targets.unsqueeze(1)
        expected = (n, actions.shape[1], actions.shape[2] - 2, *shape)
        if tuple(future_targets.shape) != expected:
            raise ValueError(f"future_targets must have shape {expected}.")
        for start in range(0, n, batch_size):
            # Validate without copying the complete future bank to the device.
            _finite(future_targets[start:start + batch_size], "future_targets")
    return actions, future_targets


@contextmanager
def _frozen(model):
    parameters = [(p, p.requires_grad) for p in model.parameters()]
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        model.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training


def _rollout(model, history, actions, shape):
    future = model.rollout(history, actions)
    expected = (history.shape[0], actions.shape[1] - 2, *shape)
    if not isinstance(future, torch.Tensor) or tuple(future.shape) != expected:
        raise ValueError(f"model.rollout must return {expected}.")
    return _finite(future, "model rollout")


def fit_editor(name, histories, requests, targets, *, rank=8, seed=0, steps=200,
               lr=1e-3, device="cpu", model=None, actions=None, future_targets=None,
               anchor_y=None, states_anchor=None, basis_samples=256, batch_size=32,
               grid=2, context_dim=16, hidden_dim=64, ridge=1e-2,
               rollout_weight=1.0) -> tuple[nn.Module, dict[str, Any]]:
    """Fit an editor using ONLY supplied training data; return it and a loss log.

    histories/targets: [N,3,*latent]; requests: [N,6] physical anchor deltas.
    inverse requires factual anchor_y [N,6] (alias states_anchor), paired with
    histories[:,0], not the rollout-entry endpoint. Scorers cannot be passed in.
    mlp and rollout start from the paired ridge solution, then take ``steps``
    Adam updates. Rollout loss adds rollout_weight times native future MSE.
    Other editors ignore model/actions/future_targets and never call a model.
    An editor returned by make_editor may replace name; it is reset/refitted
    in place using its own architecture, regularization and seed settings.
    """
    steps, batch_size = _integer(steps, "steps", 0), _integer(batch_size, "batch_size")
    basis_samples = _integer(basis_samples, "basis_samples")
    lr, rollout_weight = _real(lr, "lr", True), _real(rollout_weight, "rollout_weight")
    histories = _tensor(histories, "histories")
    if histories.ndim not in (3, 4):
        raise ValueError("histories must be [N,3,*latent_shape].")
    if isinstance(name, _HistoryEditor):
        editor = name.to(device="cpu", dtype=torch.float32)
        fresh = make_editor(editor.name, editor.latent_shape, editor.rank, editor.seed, grid=editor.grid,
                            context_dim=editor.context_dim, hidden_dim=editor.hidden_dim, ridge=editor.ridge)
        editor.load_state_dict(fresh.state_dict())
    else:
        editor = make_editor(name, histories.shape[2:], rank, seed, grid=grid,
                             context_dim=context_dim, hidden_dim=hidden_dim, ridge=ridge)
    histories, (n,) = _states(histories, (3, *editor.latent_shape), "histories", batched=True)
    targets, prefix = _states(targets, (3, *editor.latent_shape), "targets", batched=True)
    requests, request_prefix = _states(requests, (6,), "requests", batched=True)
    if prefix != (n,) or request_prefix != (n,):
        raise ValueError("histories, targets and requests must have matching N.")
    requests = _batch(requests, slice(None))
    if anchor_y is not None and states_anchor is not None:
        raise ValueError("Supply only one of anchor_y and states_anchor.")
    anchor_y = anchor_y if anchor_y is not None else states_anchor
    if editor.name == "inverse":
        if anchor_y is None:
            raise ValueError("inverse requires factual anchor_y (or states_anchor), not test labels.")
        anchor_y, anchor_prefix = _states(anchor_y, (6,), "anchor_y", batched=True)
        if anchor_prefix != (n,):
            raise ValueError("anchor_y must have matching N.")
    device = torch.empty(0, device=device).device
    if editor.name == "rollout":
        actions, future_targets = _rollout_data(model, actions, future_targets, n,
                                               editor.latent_shape, device, batch_size)

    with torch.no_grad():
        used = 0 if editor.name == "identity" else _fit_basis(editor, histories, targets, basis_samples, batch_size)
        context = None if editor.name in ("identity", "inverse") else _fit_context(editor, histories, batch_size)
        coefficients, residuals = _project_targets(editor, histories, targets, batch_size)
        design = editor._design(context, requests)
        initial = _history_mse(editor, design, coefficients, residuals, batch_size)
        if editor.name == "inverse":
            editor.probe.fit(histories[:, 0], anchor_y)
            # The derivative has no mean/intercept. Invert only the reduced
            # basis-to-anchor map, then lift with the SAME full-history basis.
            pooled = editor.probe.features(editor.basis[:, 0])
            sensitivity = ((pooled / editor.probe.input_std) @ editor.probe.weight
                           * editor.probe.output_scale / editor.request_scale)
            editor.linear.weight.copy_(torch.linalg.pinv(sensitivity.double(), rtol=1e-6).T)
        elif editor.name != "identity":
            scale = coefficients.square().mean(0).sqrt()
            editor.coefficient_scale.copy_(torch.where(scale > 1e-6, scale, torch.ones_like(scale)))
            weight, _ = _ridge(design, coefficients / editor.coefficient_scale, editor.ridge, intercept=False)
            editor.linear.weight.copy_(weight.T)
    editor.to(device)
    baseline = _history_mse(editor, design, coefficients, residuals, batch_size)
    log = {"name": editor.name, "seed": editor.seed, "rank": editor.rank,
           "effective_rank": editor.effective_rank.item(), "basis_samples": used, "n_train": n,
           "initial_history_mse": initial, "final_history_mse": baseline,
           "losses": [{"step": 0, "loss": baseline, "history_mse": baseline, "rollout_mse": 0.0}]}

    if editor.mlp is not None and steps:
        optimizer = torch.optim.Adam(editor.parameters(), lr=lr)
        generator = torch.Generator().manual_seed(editor.seed)
        frozen = _frozen(model) if editor.name == "rollout" else nullcontext()
        width = editor.basis[0].numel()
        with frozen:
            for step in range(1, steps + 1):
                index = torch.randperm(n, generator=generator)[:batch_size]
                optimizer.zero_grad(set_to_none=True)
                predicted = editor._coefficients(_batch(design, index, device))
                history_loss = ((predicted - _batch(coefficients, index, device)).square().sum(1).mean() / width
                                + _batch(residuals, index, device).mean())
                rollout_loss = 0.0
                if editor.name == "rollout" and rollout_weight:
                    edited = editor._apply_edit(_batch(histories, index, device), predicted,
                                                _batch(requests, index, device))
                    history_loss.backward(retain_graph=True)
                    native_history = _batch(targets, index, device) if future_targets is None else None
                    for candidate in range(actions.shape[1]):
                        action = _batch(actions[:, candidate], index, device)
                        with torch.no_grad():
                            native = (_rollout(model, native_history, action, editor.latent_shape)
                                      if future_targets is None else _batch(future_targets[:, candidate], index, device))
                        future = _rollout(model, edited, action, editor.latent_shape)
                        loss = F.mse_loss(future, native.detach())
                        if not loss.requires_grad:
                            raise ValueError("model.rollout must preserve gradients to edited history.")
                        # Backpropagate candidates separately; do not retain a
                        # bank of K world-model activation graphs on the GPU.
                        (loss * rollout_weight / actions.shape[1]).backward(
                            retain_graph=candidate + 1 < actions.shape[1])
                        rollout_loss += loss.detach().item() / actions.shape[1]
                else:
                    history_loss.backward()
                nn.utils.clip_grad_norm_(editor.parameters(), max_norm=10.0, error_if_nonfinite=True)
                optimizer.step()
                history_mse = history_loss.detach().item()
                log["losses"].append({"step": step, "loss": history_mse + rollout_weight * rollout_loss,
                                      "history_mse": history_mse, "rollout_mse": rollout_loss})
        log["final_history_mse"] = _history_mse(editor, design, coefficients, residuals, batch_size)
    editor.fitted.fill_(True)
    return editor.eval(), log


def matched_random(editor, history, request, generator):
    """Random coefficients in the SAME fitted basis, matched in per-history L2.

    Matches the edit's full-history Frobenius norm and available rank/location
    span, not every patch's magnitude. Rank-one controls can only flip sign.
    Zero requests/zero edits remain unchanged. No target or model is consulted.
    The explicit generator, including a CPU generator for a CUDA editor, is the
    sole randomness source.
    """
    if not isinstance(editor, _HistoryEditor) or not isinstance(generator, torch.Generator):
        raise TypeError("matched_random requires a make_editor module and torch.Generator.")
    history, request = editor._inputs(history, request)
    if editor.name == "identity" or editor.effective_rank.item() == 0:
        return history
    coefficient = editor._from_history(history, request)
    random = torch.randn(coefficient.shape, generator=generator, device=generator.device,
                         dtype=coefficient.dtype).to(coefficient.device)
    random[:, editor.effective_rank.item():] = 0
    gram = editor.basis.flatten(1) @ editor.basis.flatten(1).T
    norm = (coefficient @ gram * coefficient).sum(1, keepdim=True).clamp_min(0).sqrt()
    random_norm = (random @ gram * random).sum(1, keepdim=True).clamp_min(0).sqrt()
    random = random * norm / random_norm.clamp_min(torch.finfo(random.dtype).tiny)
    return editor._apply_edit(history, random, request)


def save_editor(path, editor):
    """Save a tensor-only, versioned checkpoint, never an arbitrary module pickle."""
    if not isinstance(editor, _HistoryEditor):
        raise TypeError("save_editor requires an editor from make_editor/fit_editor.")
    config = {"version": torch.tensor(1), "name": torch.tensor(_NAMES.index(editor.name)),
              "latent_shape": torch.tensor(editor.latent_shape), "rank": torch.tensor(editor.rank),
              "seed": torch.tensor(editor.seed), "grid": torch.tensor(editor.grid),
              "context_dim": torch.tensor(editor.context_dim), "hidden_dim": torch.tensor(editor.hidden_dim),
              "ridge": torch.tensor(editor.ridge, dtype=torch.float64)}
    state = {key: value.detach().cpu().contiguous() for key, value in editor.state_dict().items()}
    torch.save({"config": config, "state_dict": state}, Path(path))


def load_editor(path, device="cpu") -> nn.Module:
    """Reconstruct only the registered editor architecture using weights_only=True."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    keys = {"version", "name", "latent_shape", "rank", "seed", "grid", "context_dim", "hidden_dim", "ridge"}
    if (not isinstance(payload, dict) or set(payload) != {"config", "state_dict"}
            or not isinstance(payload["config"], dict) or set(payload["config"]) != keys
            or not isinstance(payload["state_dict"], dict)):
        raise ValueError("Invalid editor checkpoint schema.")
    config, state = payload["config"], payload["state_dict"]
    if (any(not isinstance(v, torch.Tensor) for v in (*config.values(), *state.values()))
            or any(not isinstance(key, str) for key in state)):
        raise ValueError("Editor checkpoints must contain tensor dictionaries only.")
    for key in keys - {"latent_shape", "ridge"}:
        if config[key].ndim != 0 or config[key].dtype != torch.int64:
            raise ValueError(f"Invalid checkpoint {key}.")
    if (config["latent_shape"].ndim != 1 or config["latent_shape"].dtype != torch.int64
            or config["ridge"].ndim != 0 or not config["ridge"].is_floating_point()
            or config["version"].item() != 1 or not 0 <= config["name"].item() < len(_NAMES)):
        raise ValueError("Invalid editor checkpoint configuration/version.")
    kwargs = {key: config[key].item() for key in ("rank", "seed", "grid", "context_dim", "hidden_dim", "ridge")}
    editor = make_editor(_NAMES[config["name"].item()], tuple(config["latent_shape"].tolist()), **kwargs)
    editor.load_state_dict(state, strict=True)
    return editor.to(device).eval()