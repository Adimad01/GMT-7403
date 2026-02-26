GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) --system-prompt ../system-prompt.txt --provider azure --model gpt-35-turbo-0125 --delay 0.5 --repeat "0:30" --seed 123 --temperature 0.0  -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
