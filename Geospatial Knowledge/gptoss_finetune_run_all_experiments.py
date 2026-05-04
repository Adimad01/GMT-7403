import subprocess
import logging

EXEC_SCRIPT = "gptoss_main.py"
P_TYPES = ['0', '1']
P_LENGTHS = ['zero-shot', '3-shot']

# UPDATE: We now tell the script to run the base model (None) 
# AND the fine-tuned model by pointing to the final adapter folder.
ADAPTERS = [
    None, 
    "finetuned_gptoss20b_geocoords/final_adapter" 
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

def main():
    """Iterate over all adapter/prompt-type/shot combinations and launch inference subprocesses."""
    log.info("Starting GPT-OSS Benchmarking Suite...")

    for adapter in ADAPTERS:
        # Just some nice logging to show which model is running
        model_name = "BASE MODEL" if adapter is None else "FINE-TUNED MODEL"
        log.info(f"\n{'='*60}")
        log.info(f"🚀 NOW BENCHMARKING: {model_name}")
        log.info(f"{'='*60}")

        for p_type in P_TYPES:
            for p_length in P_LENGTHS:
                log.info(f"\n>>> Triggering Configuration: Type {p_type}, {p_length}")
                
                cmd = ["python", EXEC_SCRIPT, "--p_type", p_type, "--p_length", p_length]
                if adapter: 
                    cmd.extend(["--adapter", adapter])
                
                try:
                    subprocess.run(cmd, check=True)
                except KeyboardInterrupt:
                    log.warning("Process interrupted by user. Exiting the master loop.")
                    return
                except Exception as e:
                    log.error(f"Failed configuration {p_type}/{p_length}: {e}")

    log.info("\n🎉 All scheduled experiments have been processed.")

if __name__ == "__main__":
    main()