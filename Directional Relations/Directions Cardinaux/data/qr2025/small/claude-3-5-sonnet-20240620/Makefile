GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) --system-prompt ../system-prompt.txt --provider anthropic --model "claude-3-5-sonnet-20240620" --delay 0.1 --temperature 0.0 --max_tokens 512 --repeat "0:30" -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
