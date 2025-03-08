#!/bin/bash

# Start Ollama in the background.
 /bin/ollama serve &
# Record Process ID.
pid=$!

# Pause for Ollama to start.
sleep 5

# IFS=', ' read -r -a models <<< "$OLLAMA_MODELS_USED"
for model in "${models[@]}"; do
    echo "📥 Pulling model: deepseek-r1:1.5b"
    ollama pull "deepseek-r1:1.5b"
    echo "🟢 deepseek-r1:1.5b downloaded!"
done

# Wait for Ollama process to finish.
wait $pid