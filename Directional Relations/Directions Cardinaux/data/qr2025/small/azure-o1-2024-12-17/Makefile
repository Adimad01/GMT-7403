GOLEM = ~/git/golem/golem.py

TEMP := $(shell date +%Y%m%d%H%M%S).jsonl

answers.jsonl: ../prompts.jsonl ../system-prompt.txt
	$(GOLEM) -v --seed 123 --system-prompt ../system-prompt.txt --provider azure --model o1-2024-12-17 --url "$$AZURE_OPENAI_ENDPOINT_4" --key "$$AZURE_OPENAI_API_KEY_4" --repeat "0:3"  -f $< > $(TEMP)
	mv $(TEMP) $@

clean:
	rm *.jsonl
