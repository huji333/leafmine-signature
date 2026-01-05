IMAGE ?= leafmine-signature:latest
DATA_DIR ?= $(CURDIR)/data
PYTHON ?= python3
SEGMENTED_DIR ?= $(DATA_DIR)/segmented
ANNOTATION_CSV ?= $(DATA_DIR)/sample_annotation.csv
SEGMENTED_GLOB ?= segmented_*.png

HOST_ARCH := $(shell /usr/bin/uname -m 2>/dev/null)
DEFAULT_PLATFORM :=
DEFAULT_BUILD_CMD := docker build

ifneq ($(filter arm64 aarch64,$(HOST_ARCH)),)
  DEFAULT_PLATFORM := linux/amd64
  DEFAULT_BUILD_CMD := docker buildx build --load
endif

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
	$(PYTHON) scripts/generate_sample_annotation.py --input-dir "$(SEGMENTED_DIR)" --output "$(ANNOTATION_CSV)" --glob "$(SEGMENTED_GLOB)"

remove_data:
	@mkdir -p $(DATA_DIR)
	@echo "Pruning generated artifacts under $(DATA_DIR) (set PURGE=1 to remove segmented/ and signatures/ too)."
	@set -e; \
	for entry in $$(/bin/ls -1A $(DATA_DIR) 2>/dev/null); do \
		case $$entry in \
			segmented|signatures) \
				if [ "$$PURGE" = "1" ]; then \
					echo "Removing $(DATA_DIR)/$$entry"; \
					/bin/rm -rf "$(DATA_DIR)/$$entry"; \
				else \
					echo "Preserving $(DATA_DIR)/$$entry"; \
					continue; \
				fi ;; \
			*) \
				echo "Removing $(DATA_DIR)/$$entry"; \
				/bin/rm -rf "$(DATA_DIR)/$$entry"; \
			;; \
		esac; \
	done; \
	if [ "$$PURGE" != "1" ]; then \
		/bin/mkdir -p "$(DATA_DIR)/skeletonized" "$(DATA_DIR)/tmp"; \
	fi

clean:
	docker image rm $(IMAGE) || true
