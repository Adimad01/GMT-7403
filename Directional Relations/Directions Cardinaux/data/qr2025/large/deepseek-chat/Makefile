GOLEM = ~/git/golem/golem.py
SCRIPTS=../../../../scripts

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) --provider deepseek --model deepseek-chat --system-prompt ../system-prompt.txt --url "$$DEEPSEEK_ENDPOINT" --key "$$DEEPSEEK_API_KEY" --repeat "0:3"  --temperature 0.0 --seed 123 -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm answers.jsonl
