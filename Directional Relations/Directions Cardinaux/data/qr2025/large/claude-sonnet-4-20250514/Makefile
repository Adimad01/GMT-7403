GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --provider anthropic --system-prompt ../system-prompt.txt --model "claude-sonnet-4-20250514" --temperature 0.0 --max_tokens 1024 -f ../prompts.jsonl > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
