import subprocess
import logging

EXEC_SCRIPT = "gptoss_main.py"
P_TYPES = ['0', '1']
P_LENGTHS = ['zero-shot', '3-shot']
ADAPTERS = [None] # Add paths here if testing LoRAs

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

def main():
    """Launch gptoss_main.py subprocesses for all prompt-type and shot-length combinations."""
    log.info("Starting GPT-OSS Benchmarking Suite...")
    
    for adapter in ADAPTERS:
        for p_type in P_TYPES:
            for p_length in P_LENGTHS:
                log.info(f"\n>>> Triggering Configuration: Type {p_type}, {p_length}")
                cmd = ["python", EXEC_SCRIPT, "--p_type", p_type, "--p_length", p_length]
                if adapter: cmd.extend(["--adapter", adapter])
                
                try:
                    subprocess.run(cmd, check=True)
                except KeyboardInterrupt:
                    log.warning("Process interrupted by user. Exiting the master loop.")
                    return
                except Exception as e:
                    log.error(f"Failed configuration {p_type}/{p_length}: {e}")

    log.info("All scheduled experiments have been processed.")

if __name__ == "__main__":
    main()