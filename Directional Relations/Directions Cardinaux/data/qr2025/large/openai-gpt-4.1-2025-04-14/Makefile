GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --seed 123 --system-prompt ../system-prompt.txt --provider openai --model gpt-4.1 --temperature 0  --repeat "0:3"  -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
