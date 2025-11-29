"""
Remote Simulation Runner for E2B

This script packages the simulation and runs it on E2B sandboxes.
E2B provides isolated Python environments perfect for CPU-bound simulations.

Usage:
    # Run all strategies in parallel on E2B
    python -m backend.simulate_remote --platform e2b

    # Run on Modal instead
    python -m backend.simulate_remote --platform modal
"""

import argparse
import json
import sys
import os

# Platform-specific imports will be done conditionally


def run_on_e2b(
    strategy: str,
    num_images: int,
    max_labels: int,
    output: str = None,
) -> dict:
    """
    Run simulation on E2B sandbox.

    E2B provides secure, isolated Python sandboxes perfect for running
    untrusted code or CPU-intensive simulations without local resource usage.
    """
    try:
        from e2b_code_interpreter import Sandbox
    except ImportError:
        print("ERROR: e2b_code_interpreter not installed.")
        print("Install with: pip install e2b-code-interpreter")
        sys.exit(1)

    print(f"Starting E2B sandbox for strategy: {strategy}")

    # Create sandbox
    with Sandbox() as sandbox:
        print("Sandbox created. Installing dependencies...")

        # Install dependencies
        sandbox.run_code("""
import subprocess
import sys

packages = [
    'torch',
    'torchvision', 
    'numpy',
    'pillow',
    'qdrant-client',
]

for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("Dependencies installed!")
""")

        # Upload simulation code
        print("Uploading simulation code...")

        # Read local files to upload
        current_dir = os.path.dirname(os.path.abspath(__file__))

        files_to_upload = [
            "app/common.py",
            "app/oracle.py",
            "simulate.py",
        ]

        for file_path in files_to_upload:
            full_path = os.path.join(current_dir, file_path)
            with open(full_path, "r") as f:
                content = f.read()

            # Create directory structure in sandbox
            remote_path = f"/home/user/backend/{file_path}"
            remote_dir = os.path.dirname(remote_path)

            sandbox.run_code(f"""
import os
os.makedirs('{remote_dir}', exist_ok=True)

content = '''{content}'''

with open('{remote_path}', 'w') as f:
    f.write(content)
""")

        print("Running simulation...")

        # Run simulation
        execution = sandbox.run_code(f"""
import sys
sys.path.insert(0, '/home/user')

from backend.simulate import SimulationRunner

sim = SimulationRunner(
    strategy_name='{strategy}',
    num_images={num_images},
    use_qdrant_memory=True,
)

metrics = sim.run(max_labels={max_labels if max_labels else "None"})

# Print results as JSON for parsing
import json
print("RESULTS_START")
print(json.dumps(metrics))
print("RESULTS_END")
""")

        # Parse results
        output_text = execution.text

        if "RESULTS_START" in output_text:
            start = output_text.index("RESULTS_START") + len("RESULTS_START")
            end = output_text.index("RESULTS_END")
            results_json = output_text[start:end].strip()
            results = json.loads(results_json)

            print(f"\nSimulation complete!")
            print(f"Labels: {results['final_state']['num_labeled']}")
            print(f"Accuracy: {results['final_state']['accuracy']:.3f}")

            if output:
                with open(output, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to: {output}")

            return results
        else:
            print("ERROR: Could not parse results from sandbox output")
            print(output_text)
            return None


def run_on_modal(
    strategy: str,
    num_images: int,
    max_labels: int,
    output: str = None,
) -> dict:
    """
    Run simulation on Modal.

    Modal provides serverless compute with easy Python deployment.
    Good for parallel execution and CPU/GPU flexibility.
    """
    import modal

    # Define Modal app
    app = modal.App(f"quicksort-sim-{strategy}")

    # Define image with dependencies
    image = modal.Image.debian_slim().pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "qdrant-client",
    )

    # Mount local backend code
    backend_mount = modal.Mount.from_local_dir(
        os.path.join(os.path.dirname(__file__)), remote_path="/root/backend"
    )

    @app.function(
        image=image,
        mounts=[backend_mount],
        cpu=4,  # CPU-only since we're using Qdrant in-memory
        timeout=3600,  # 1 hour max
    )
    def run_simulation(strategy: str, num_images: int, max_labels: int):
        import sys

        sys.path.insert(0, "/root")

        from backend.simulate import SimulationRunner

        print(f"Running simulation for {strategy} on Modal...")

        sim = SimulationRunner(
            strategy_name=strategy,
            num_images=num_images,
            use_qdrant_memory=True,
        )

        metrics = sim.run(max_labels=max_labels)
        return metrics

    # Deploy and run
    print(f"Deploying to Modal: {strategy}")

    with app.run():
        result = run_simulation.remote(strategy, num_images, max_labels)

    print(f"\nSimulation complete!")
    print(f"Labels: {result['final_state']['num_labeled']}")
    print(f"Accuracy: {result['final_state']['accuracy']:.3f}")

    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {output}")

    return result


def run_parallel_e2b(strategies: list, num_images: int, max_labels: int):
    """Run multiple strategies in parallel on E2B."""
    import concurrent.futures

    print(f"Running {len(strategies)} strategies in parallel on E2B...")

    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(strategies)) as executor:
        future_to_strategy = {
            executor.submit(
                run_on_e2b, strategy, num_images, max_labels, f"results_{strategy}.json"
            ): strategy
            for strategy in strategies
        }

        for future in concurrent.futures.as_completed(future_to_strategy):
            strategy = future_to_strategy[future]
            try:
                result = future.result()
                results[strategy] = result
                print(f"✓ {strategy} complete")
            except Exception as e:
                print(f"✗ {strategy} failed: {e}")

    # Print comparison
    print(f"\n{'=' * 60}")
    print("RESULTS COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Strategy':<20} {'Labels':<10} {'Accuracy':<12} {'Time (s)':<12}")
    print(f"{'-' * 60}")

    for strategy, metrics in results.items():
        if metrics:
            final = metrics["final_state"]
            print(
                f"{strategy:<20} "
                f"{final['num_labeled']:<10} "
                f"{final['accuracy']:<12.3f} "
                f"{metrics['total_time']:<12.2f}"
            )

    # Save combined results
    with open("results_all.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCombined results saved to: results_all.json")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run QuickSort simulations on remote compute (E2B or Modal)"
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=["e2b", "modal"],
        default="e2b",
        help="Platform to run on (default: e2b)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        help="Strategy to simulate (default: all)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1000,
        help="Number of images (default: 1000)",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Max labels to collect (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output file (default: results.json)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run strategies in parallel (E2B only)",
    )

    args = parser.parse_args()

    # Import strategy registry
    from backend.app.oracle import STRATEGIES

    if args.strategy == "all":
        strategies = list(STRATEGIES.keys())
    else:
        strategies = [args.strategy]

    if args.platform == "e2b":
        if args.parallel and len(strategies) > 1:
            run_parallel_e2b(strategies, args.num_images, args.max_labels)
        else:
            for strategy in strategies:
                run_on_e2b(strategy, args.num_images, args.max_labels, args.output)

    elif args.platform == "modal":
        for strategy in strategies:
            run_on_modal(strategy, args.num_images, args.max_labels, args.output)


if __name__ == "__main__":
    main()
