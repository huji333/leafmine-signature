IMAGE ?= leafmine-signature:latest

.PHONY: setup build run clean

setup:
	$(MAKE) build

build:
	docker build -t $(IMAGE) .

run: build
	docker run --rm -it -p 7860:7860 $(IMAGE)

clean:
	docker image rm $(IMAGE) || true
