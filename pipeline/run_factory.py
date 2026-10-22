import subprocess
import sys

steps = [
    "pipeline/ingest.py",
    "pipeline/quality.py",
    "pipeline/cluster.py",
    "pipeline/label_clusters.py",
    "pipeline/extract.py"
]

print("\n================================")
print("STARTING KNOWLEDGE FACTORY")
print("================================")

for step in steps:

    print(f"\nRunning: {step}")

    result = subprocess.run(
        [sys.executable, step]
    )

    if result.returncode != 0:
        print(f"\nFAILED: {step}")
        sys.exit(1)

    print(f"COMPLETED: {step}")


print("\n================================")
print("KNOWLEDGE FACTORY COMPLETED")
print("================================")