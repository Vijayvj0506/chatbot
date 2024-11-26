import os
import time
import requests
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from queue import Queue, Empty
from threading import Thread
import json
import logging

# Initialize logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Hugging Face API key (replace with your own API key)
HF_API_KEY = os.getenv("HF_API_KEY")  # Ensure the key is set in your environment variables
HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"


# Queue for handling API requests
REQUEST_QUEUE = Queue()
MAX_RETRIES = 5
INITIAL_DELAY = 2  # Start with 2 seconds
BACKOFF_MULTIPLIER = 2  # Exponential backoff factor
MAX_BACKOFF = 60  # Cap maximum delay at 60 seconds
MAX_TIMEOUT = 90  # Maximum response queue wait time


def process_requests():
    """Background thread to handle API requests."""
    while True:
        try:
            # Retrieve the message and the response queue
            user_message, response_queue = REQUEST_QUEUE.get()
            retries = 0
            delay = INITIAL_DELAY

            while retries < MAX_RETRIES:
                try:
                    # Request to Hugging Face API
                    response = requests.post(
                        HF_API_URL,
                        headers={"Authorization": f"Bearer {HF_API_KEY}"},
                        json={"inputs": user_message},
                    )

                    # If model is loading, wait and retry
                    if response.status_code == 503 and "currently loading" in response.text:
                        estimated_time = response.json().get("estimated_time", delay)
                        logger.error(
                            "Model is loading, retrying after %f seconds.", estimated_time
                        )
                        time.sleep(min(estimated_time, MAX_BACKOFF))
                        retries += 1
                        continue

                    # Check if the response is valid
                    if response.status_code == 200:
                        try:
                            bot_response = response.json()[0].get('generated_text', '')
                            if bot_response:
                                response_queue.put(bot_response)
                            else:
                                logger.error("Empty 'generated_text' in response.")
                                response_queue.put("Error: No valid response from model.")
                        except KeyError as e:
                            logger.error(f"Key error in response: {e}")
                            response_queue.put("Error: Invalid response format.")
                        break

                    elif response.status_code == 401:
                        logger.error("Hugging Face API error: Unauthorized. Check your API key.")
                        response_queue.put("Error: Unauthorized. Check your API key.")
                        break
                    else:
                        logger.error("Hugging Face API error: %s", response.text)
                        response_queue.put(f"Error: Hugging Face API error: {response.text}")
                        break

                except requests.exceptions.RequestException as e:
                    retries += 1
                    logger.error(
                        "Request error: %s. Retry %d/%d after %d seconds.",
                        str(e), retries, MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * BACKOFF_MULTIPLIER, MAX_BACKOFF)
                except Exception as e:
                    retries += 1
                    logger.error(
                        "Unexpected error: %s. Retry %d/%d after %d seconds.",
                        str(e), retries, MAX_RETRIES, delay,
                        exc_info=True,
                    )
                    time.sleep(delay)
                    delay = min(delay * BACKOFF_MULTIPLIER, MAX_BACKOFF)

            if retries == MAX_RETRIES:
                logger.error("Failed to process request after %d retries.", MAX_RETRIES)
                response_queue.put(
                    "Error: Unable to process your request after multiple attempts. Please try again later."
                )

        except Exception as e:
            logger.error("Unexpected error in process_requests: %s", str(e), exc_info=True)
        finally:
            REQUEST_QUEUE.task_done()


# Start the background thread for request processing
Thread(target=process_requests, daemon=True).start()


def index(request):
    """Render the chat page."""
    return render(request, 'chat.html')


@csrf_exempt
def chat_view(request):
    """API endpoint for chatbot interaction."""
    if request.method == 'POST':
        try:
            # Parse JSON payload
            data = json.loads(request.body.decode('utf-8'))
            user_message = data.get('message', '').strip()

            if not user_message:
                return JsonResponse({'error': 'Message cannot be empty'}, status=400)

            # Prepare response queue for this request
            response_queue = Queue()
            REQUEST_QUEUE.put((user_message, response_queue))

            # Wait for response with a timeout
            try:
                bot_response = response_queue.get(timeout=MAX_TIMEOUT)
                return JsonResponse({'response': bot_response})
            except Empty:
                logger.error("Response queue timed out after %d seconds.", MAX_TIMEOUT)
                return JsonResponse(
                    {'error': 'The request timed out. Please try again later.'},
                    status=500,
                )

        except json.JSONDecodeError:
            logger.error("Invalid JSON payload received.", exc_info=True)
            return JsonResponse({'error': 'Invalid JSON payload'}, status=400)
        except Exception as e:
            logger.error("Unexpected error occurred: %s", str(e), exc_info=True)
            return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Only POST method is allowed.'}, status=405)
