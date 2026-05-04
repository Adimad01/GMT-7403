GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) --system-prompt ../system-prompt.txt --provider azure --model gpt-4o-mini-2024-07-18 --delay 0.5 --temperature 0.0 --seed 123 --repeat "0:30"  -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
