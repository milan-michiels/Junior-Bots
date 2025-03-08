#!/bin/bash

# Start Ollama in the background.
 /bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

IFS=', ' read -r -a models <<< "$OLLAMA_MODELS_USED"
for model in "${models[@]}"; do
    echo "📥 Pulling model: $model"
    ollama pull "$model"
    echo "🟢 $model downloaded!"
done

# Wait for Ollama process to finish.
wait $pid