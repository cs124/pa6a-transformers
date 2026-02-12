import torch
import numpy as np
from perplexity import compute_perplexity

def test_uniform_logits_gives_vocab_size():
    """When all logits are equal, every token is equally likely.
    Perplexity should equal the vocabulary size."""
    V, T = 100, 5
    logits = torch.zeros(1, T, V)
    y_true = torch.randint(0, V, (1, T))
    ppl = compute_perplexity(logits, y_true)
    assert abs(ppl - V) < 1e-3, f"Expected {V}, got {ppl}"

def test_confident_model_gives_perplexity_near_one():
    """When the model assigns overwhelming probability to the correct token,
    perplexity should be approximately 1."""
    V, T = 50, 4
    y_true = torch.tensor([[3, 7, 12, 1]])
    logits = torch.zeros(1, T, V)
    for t in range(T):
        logits[0, t, y_true[0, t]] = 100.0  # huge logit for correct token
    ppl = compute_perplexity(logits, y_true)
    assert abs(ppl - 1.0) < 1e-3, f"Expected ~1.0, got {ppl}"

def test_single_token_position():
    """With T=1, perplexity reduces to exp(-log_softmax(logit_of_target)).
    Hand-computable: logits [2.0, 1.0, 0.0], target=0 →
    log_softmax(0) = 2.0 - log(e^2 + e^1 + e^0) ≈ 2.0 - 2.4076 = -0.4076
    ppl = exp(0.4076) ≈ 1.5033"""
    logits = torch.tensor([[[2.0, 1.0, 0.0]]])  # [1, 1, 3]
    y_true = torch.tensor([[0]])                  # [1, 1]
    ppl = compute_perplexity(logits, y_true)
    expected = np.exp(-torch.log_softmax(logits, dim=-1)[0, 0, 0].item())
    assert abs(ppl - expected) < 1e-3, f"Expected {expected:.4f}, got {ppl}"

def test_two_positions_averaged_correctly():
    """With T=2 and different logits per position, perplexity should be
    exp(mean of the two per-position NLLs), not exp(sum)."""
    V = 4
    logits = torch.tensor([[[1.0, 2.0, 0.0, 0.0],
                             [0.0, 0.0, 3.0, 1.0]]])  # [1, 2, 4]
    y_true = torch.tensor([[1, 2]])  # [1, 2]
    log_probs = torch.log_softmax(logits, dim=-1)
    nll_0 = -log_probs[0, 0, 1].item()
    nll_1 = -log_probs[0, 1, 2].item()
    expected = np.exp((nll_0 + nll_1) / 2)
    ppl = compute_perplexity(logits, y_true)
    assert abs(ppl - expected) < 1e-3, f"Expected {expected:.4f}, got {ppl}"

def test_perplexity_is_always_at_least_one():
    """Perplexity is always >= 1 for any valid probability distribution."""
    V, T = 20, 6
    torch.manual_seed(42)
    logits = torch.randn(1, T, V)
    y_true = torch.randint(0, V, (1, T))
    ppl = compute_perplexity(logits, y_true)
    assert ppl >= 1.0 - 1e-6, f"Perplexity should be >= 1, got {ppl}"

def test_target_token_identity_matters():
    """Changing which token is the target should change the perplexity."""
    V = 5
    logits = torch.tensor([[[2.0, 0.0, 0.0, 0.0, 0.0]]])  # [1, 1, 5]
    ppl_correct = compute_perplexity(logits, torch.tensor([[0]]))  # high-prob target
    ppl_wrong   = compute_perplexity(logits, torch.tensor([[4]]))  # low-prob target
    assert ppl_correct < ppl_wrong, \
        f"Targeting the high-logit token ({ppl_correct}) should give lower ppl than a low-logit token ({ppl_wrong})"

def test_returns_float_not_tensor():
    """compute_perplexity should return a plain Python float, not a tensor."""
    logits = torch.zeros(1, 3, 10)
    y_true = torch.zeros(1, 3, dtype=torch.long)
    ppl = compute_perplexity(logits, y_true)
    assert isinstance(ppl, (float, int, np.floating)), \
        f"Expected a float, got {type(ppl)}"


if __name__ == "__main__":
    tests = [
        ("Uniform logits → ppl equals vocab size",       test_uniform_logits_gives_vocab_size),
        ("Confident model → ppl ≈ 1",                    test_confident_model_gives_perplexity_near_one),
        ("Single position hand-computed value",           test_single_token_position),
        ("Two positions averaged (not summed)",           test_two_positions_averaged_correctly),
        ("Perplexity is always ≥ 1",                     test_perplexity_is_always_at_least_one),
        ("Different target tokens → different ppl",       test_target_token_identity_matters),
        ("Returns a float, not a tensor",                 test_returns_float_not_tensor),
    ]
    all_passed = True
    for name, fn in tests:
        try:
            fn()
            print(f"  PASSED: {name}")
        except AssertionError as e:
            print(f"  FAILED: {name} — {e}")
            all_passed = False
        except Exception as e:
            print(f"  ERROR:  {name} — {type(e).__name__}: {e}")
            all_passed = False
    print()
    print("All tests passed!" if all_passed else "Some tests failed.")
