"""Reuse the audited scorer, identifying the prior published release correctly.

Arguments are identical to reproduce_order_robot.py. Each run starts a fresh
solver process; no development cache is supplied to the evaluated solver.
"""
import reproduce_order_robot as harness

harness.BASELINE_COMMIT = "991a49b48092286967b5bd7cada465e336ebc4af"

if __name__ == "__main__":
    harness.main()
