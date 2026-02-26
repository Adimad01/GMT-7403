GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --system-prompt ../system-prompt.txt --provider google --model gemini-1.5-flash-001 --delay 1 --seed 123 --temperature 0.0 --repeat "0:30" -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
