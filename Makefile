IMAGE ?= leafmine-signature:latest

.PHONY: setup build run clean

setup:
	uv lock
	$(MAKE) build

build:
	docker build -t $(IMAGE) .

run:
	@docker image inspect $(IMAGE) >/dev/null 2>&1 || $(MAKE) setup
	docker run --rm -it -p 7860:7860 $(IMAGE)

clean:
	docker image rm $(IMAGE) || true
