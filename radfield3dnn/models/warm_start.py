"""Starting a run from another run's weights.

Only the weights travel: the optimizer, the schedule and the epoch counter start fresh, so a warm-started
run differs from a cold one in its initial point and nothing else.
"""
import torch


def load_warm_start(model, checkpoint_path: str) -> dict:
    """Load a Lightning checkpoint's weights into `model` and say what came from where.

    Every tensor in the checkpoint must find a parameter of the same name and shape in `model`, and every
    parameter of `model` the checkpoint lacks must be one the model declares new
    (`model.warm_start_new_parameters`, name prefixes: the head a subclass adds). Anything else means the two
    runs are not the same network — a different width, encoder or conditioning — and silently training the
    mismatch from a half-matching start would answer a different question than the one asked.

    Returns ``{"loaded": tensors copied, "new": names left at their fresh initialisation}``.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    # a shape mismatch raises here even with strict=False
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"{checkpoint_path} holds {len(result.unexpected_keys)} tensors this model has no "
                         f"parameter for, e.g. {result.unexpected_keys[:3]}: it is a different network.")
    declared = tuple(model.warm_start_new_parameters)
    new = [k for k in result.missing_keys if declared and k.startswith(declared)]
    undeclared = [k for k in result.missing_keys if k not in new]
    if undeclared:
        raise ValueError(f"{len(undeclared)} parameters of this model are not in {checkpoint_path} and are not "
                         f"declared new ({declared or 'none'}), e.g. {undeclared[:3]}.")
    return {"loaded": len(state), "new": new}
