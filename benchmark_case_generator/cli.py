"""Command-line interface for benchmark case generator."""

import argparse
import sys
from pathlib import Path

from .client import ClientConfig, ModelClient
from .diversity import DiversityTracker
from .storage import GeneratorState, OutputManager
from .generation import CaseGenerator, DEFAULT_CONCLUSION_DISTRIBUTION
from .models import ExpectedConclusion, Difficulty


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate software-engineering benchmark test cases using Qwen models."
    )
    
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of test cases to generate (default: 1)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./generated_tests",
        help="Output directory for generated .md files (default: ./generated_tests)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="qwen",
        help="Model identifier (default: qwen)"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:1234/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:1234/v1)"
    )
    
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Environment variable name for API key (optional)"
    )
    
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated list of languages to use (default: all supported)"
    )
    
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default=None,
        help="Fix difficulty level for all cases"
    )
    
    parser.add_argument(
        "--bug-ratio",
        type=float,
        default=0.40,
        help="Ratio of BUG cases (default: 0.40)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per case (default: 5)"
    )
    
    parser.add_argument(
        "--state-file",
        type=str,
        default="generator_state.json",
        help="Path to state manifest file (default: generator_state.json)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate without writing files (for testing)"
    )
    
    parser.add_argument(
        "--check-health",
        action="store_true",
        help="Check if the model endpoint is reachable and exit"
    )
    
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Generate 3 smoke-test cases and verify diversity"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Parse languages
    languages = None
    if args.languages:
        languages = [l.strip() for l in args.languages.split(",")]
    
    # Parse difficulty
    difficulty = None
    if args.difficulty:
        difficulty = Difficulty(args.difficulty)
    
    # Build conclusion distribution
    distribution = DEFAULT_CONCLUSION_DISTRIBUTION.copy()
    # Adjust bug ratio while keeping relative proportions of others
    non_bug_total = 1.0 - DEFAULT_CONCLUSION_DISTRIBUTION[ExpectedConclusion.BUG]
    new_non_bug_total = 1.0 - args.bug_ratio
    
    for conclusion in ExpectedConclusion:
        if conclusion != ExpectedConclusion.BUG:
            old_prob = DEFAULT_CONCLUSION_DISTRIBUTION[conclusion]
            distribution[conclusion] = (old_prob / non_bug_total) * new_non_bug_total
    
    distribution[ExpectedConclusion.BUG] = args.bug_ratio
    
    # Create client config
    config = ClientConfig(
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
    )
    
    client = ModelClient(config)
    
    # Health check
    if args.check_health:
        if client.check_health():
            print(f"✓ Model endpoint {args.base_url} is reachable")
            print(f"  Model: {args.model}")
            sys.exit(0)
        else:
            print(f"✗ Model endpoint {args.base_url} is not reachable")
            sys.exit(1)
    
    # Initialize storage
    state = GeneratorState(args.state_file)
    existing_cases = state.load()
    
    output = OutputManager(args.output)
    output.initialize()
    
    tracker = DiversityTracker()
    tracker.load_cases(existing_cases)
    
    # Initialize state if new
    if not state.model_identifier:
        state.initialize(args.model)
        state.save()
    
    # Smoke test
    if args.smoke_test:
        print("Running smoke test: generating 3 cases...")
        args.count = 3
        args.dry_run = False
    
    # Create generator
    generator = CaseGenerator(
        client=client,
        diversity_tracker=tracker,
        output_manager=output,
        state=state,
        conclusion_distribution=distribution,
        max_retries=args.max_retries,
        languages=languages,
        difficulty=difficulty,
    )
    
    # Generate cases
    starting_count = tracker.case_count
    generated = generator.generate_cases(args.count, dry_run=args.dry_run)
    
    # Report
    print("\n" + "=" * 60)
    print("GENERATION REPORT")
    print("=" * 60)
    print(f"Starting case count: {starting_count}")
    print(f"Successfully generated: {len(generated)}")
    print(f"Total case count: {tracker.case_count}")
    print(f"Output directory: {args.output}")
    print(f"State file: {args.state_file}")
    
    if generated:
        print("\nGenerated cases:")
        for case in generated:
            print(f"  - {case.filename}: {case.plan.primary_concept}")
        
        # Verify diversity
        concepts = [c.plan.primary_concept for c in generated]
        unique_concepts = set(concepts)
        if len(unique_concepts) == len(concepts):
            print("\n✓ All generated cases have unique primary concepts")
        else:
            print(f"\n⚠ Warning: {len(concepts) - len(unique_concepts)} concept collisions detected")
    
    if args.smoke_test:
        print("\nSMOKE TEST RESULTS:")
        if len(generated) == 3:
            print("✓ 3 cases were created")
            
            concepts = [c.plan.primary_concept for c in generated]
            if len(set(concepts)) == 3:
                print("✓ All three have different primary concepts")
            else:
                print("✗ Primary concepts are not all distinct")
            
            # Check markdown files don't contain ground truth
            for case in generated:
                if "expected_conclusion" not in case.markdown_content.lower():
                    print(f"✓ {case.filename} contains no ground truth")
                else:
                    print(f"✗ {case.filename} may leak ground truth")
            
            # Check state manifest
            print(f"✓ State manifest contains {state.case_count} cases with private conclusions")
        else:
            print(f"✗ Expected 3 cases, got {len(generated)}")
    
    if args.dry_run:
        print("\n(Dry run - no files were written)")
    
    sys.exit(0 if generated else 1)


if __name__ == "__main__":
    main()
