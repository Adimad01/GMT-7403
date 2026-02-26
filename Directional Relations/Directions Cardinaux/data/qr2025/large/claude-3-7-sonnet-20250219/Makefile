GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --system-prompt ../system-prompt.txt --provider anthropic --model "claude-3-7-sonnet-20250219" --temperature 0.0 --max_tokens 512 --seed 123 --repeat "0:2" -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
