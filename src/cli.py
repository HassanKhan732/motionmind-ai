"""
MotionMind AI - Unified Command Line Interface (CLI)
Provides single-command entry points to tracking, training, inspection, benchmarking, and demo execution.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(
        prog="motionmind",
        description="MotionMind AI - Intelligent Gesture HCI & Perception Toolkit"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # demo subcommand
    demo_p = subparsers.add_parser("demo", help="Run Level 2 interactive demo")
    demo_p.add_argument("--webcam", action="store_true", help="Run with live webcam feed")
    demo_p.add_argument("--simulate", action="store_true", help="Run simulation loop")
    demo_p.add_argument("--cycles", type=int, default=1, help="Simulation cycles")

    # track subcommand
    track_p = subparsers.add_parser("track", help="Start real-time hand tracker & 3D WebSocket server")

    # inspect subcommand
    inspect_p = subparsers.add_parser("inspect", help="Inspect dataset distribution and health")
    inspect_p.add_argument("--path", default="data/gestures.csv", help="Path to CSV dataset")

    # benchmark subcommand
    bench_p = subparsers.add_parser("benchmark", help="Run inference latency and FPS benchmark")
    bench_p.add_argument("--iterations", type=int, default=100, help="Benchmark sample count")

    # robustness subcommand
    rob_p = subparsers.add_parser("robustness", help="Run perturbation and noise robustness stress test")

    # psl subcommand
    psl_p = subparsers.add_parser("psl", help="List Pakistani Sign Language vocabulary and curriculum")

    args = parser.parse_args()

    if args.command == "demo":
        from demo_level2 import run_simulation_demo
        if args.webcam:
            from src.tracking.hand_tracker import main as run_cam
            run_cam()
        else:
            run_simulation_demo(num_cycles=args.cycles, delay_sec=0.15, stream_ws=True)

    elif args.command == "track":
        from src.tracking.hand_tracker import main as run_cam
        run_cam()

    elif args.command == "inspect":
        from src.ml.dataset_inspector import inspect_dataset, print_inspection_report
        rep = inspect_dataset(args.path)
        print_inspection_report(rep)

    elif args.command == "benchmark":
        from scripts.benchmark_latency import benchmark_pipeline, print_benchmark_report
        rep = benchmark_pipeline(num_iterations=args.iterations)
        print_benchmark_report(rep)

    elif args.command == "robustness":
        from src.ml.evaluate_robustness import run_robustness_suite, print_robustness_summary
        rep = run_robustness_suite()
        print_robustness_summary(rep)

    elif args.command == "psl":
        from src.ml.psl_gestures import PSL_STATIC_DICTIONARY
        print("=" * 60)
        print(" PAKISTANI SIGN LANGUAGE (PSL) VOCABULARY")
        print("=" * 60)
        for key, info in PSL_STATIC_DICTIONARY.items():
            print(f" {key:<14} | {info['urdu_name']:<24} | {info['english']}")
        print("=" * 60)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
