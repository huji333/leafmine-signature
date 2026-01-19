IMAGE ?= leafmine-signature:latest
DATA_DIR ?= $(CURDIR)/data
PYTHON ?= python3
SEGMENTED_DIR ?= $(DATA_DIR)/segmented
ANNOTATION_CSV ?= $(DATA_DIR)/sample_annotation.csv
SEGMENTED_GLOB ?= segmented_*.png

DEFAULT_PLATFORM := linux/amd64
DEFAULT_BUILD_CMD := docker buildx build --load

BUILD_PLATFORM ?= $(DEFAULT_PLATFORM)
RUN_PLATFORM ?= $(DEFAULT_PLATFORM)
BUILD_CMD ?= $(DEFAULT_BUILD_CMD)

ifneq ($(strip $(BUILD_PLATFORM)),)
  BUILD_PLATFORM_FLAG := --platform $(BUILD_PLATFORM)
endif
ifneq ($(strip $(RUN_PLATFORM)),)
  RUN_PLATFORM_FLAG := --platform $(RUN_PLATFORM)
endif

.PHONY: setup build run lint remove_data clean annotation_csv

setup:
	$(MAKE) build

build:
	$(BUILD_CMD) $(BUILD_PLATFORM_FLAG) -t $(IMAGE) .

run: build
	@mkdir -p $(DATA_DIR)
	docker run --rm -it $(RUN_PLATFORM_FLAG) -p 7860:7860 \
		-v $(DATA_DIR):/app/data \
		$(IMAGE)

lint:
ifneq ($(BUILD),)
	$(MAKE) build
endif
	docker run --rm $(RUN_PLATFORM_FLAG) \
		-e UV_CACHE_DIR=.uv-cache \
		-v $(CURDIR):/app \
		-w /app \
		$(IMAGE) \
		uv run --frozen ruff check .

annotation_csv:
	@mkdir -p $(DATA_DIR)
	docker run --rm $(RUN_PLATFORM_FLAG) \
		-e UV_CACHE_DIR=.uv-cache \
		-v $(CURDIR):/app \
		-w /app \
		$(IMAGE) \
		uv run --frozen --no-sync python scripts/generate_sample_annotation.py \
			--glob "$(SEGMENTED_GLOB)"

remove_data:
	@ANNOTATION_CSV="$(ANNOTATION_CSV)" scripts/remove_data.sh "$(DATA_DIR)"

clean:
	docker image rm $(IMAGE) || true
