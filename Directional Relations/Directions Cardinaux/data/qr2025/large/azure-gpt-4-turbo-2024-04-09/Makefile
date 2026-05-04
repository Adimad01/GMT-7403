GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) --system-prompt ../system-prompt.txt --provider azure --model "gpt-4-turbo-2024-04-09" --delay 1 --temperature 0.0 --seed 123 --repeat "0:30" -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
