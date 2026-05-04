GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) --system-prompt ../system-prompt.txt --provider google --model "gemini-1.5-pro-001" --delay 10 --temperature 0.0 --repeat "0:31" -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
