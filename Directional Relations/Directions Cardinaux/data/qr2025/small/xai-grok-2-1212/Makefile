GOLEM = ~/git/golem/golem.py
SCRIPTS=../../../../scripts

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --provider xai --model grok-2-1212 --system-prompt ../system-prompt.txt --url "$$XAI_ENDPOINT" --key "$$XAI_API_KEY" --repeat "0:3"  --temperature 0.0 --seed 123 -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm answers.jsonl
