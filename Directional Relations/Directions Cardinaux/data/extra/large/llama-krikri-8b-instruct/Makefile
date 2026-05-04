GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl.xz: answers.jsonl
	xz -k answers.jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --system-prompt ../system-prompt.txt --provider ollama --model "ilsp/llama-krikri-8b-instruct"  --temperature 0.0 --seed 123 -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm answers.jsonl answers.jsonl.xz
