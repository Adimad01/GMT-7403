GOLEM = ~/git/golem/golem.py
SCRIPTS=../../../../scripts

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --provider deepseek --model deepseek-reasoner --system-prompt ../system-prompt.txt --url "$$DEEPSEEK_ENDPOINT" --key "$$DEEPSEEK_API_KEY" --temperature 0.0 --seed 123 -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm answers.jsonl
