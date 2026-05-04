GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) --system-prompt ../system-prompt.txt --provider azureai --delay 0.5 --temperature 0.0 --seed 123 --repeat "0:30"  -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
