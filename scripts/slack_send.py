import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Read request
with open("slack-request.txt", "r") as f:
    lines = f.read().strip().splitlines()

fields = {}
for line in lines:
    if ": " in line:
        key, value = line.split(": ", 1)
        fields[key.strip()] = value.strip()

channel = fields.get("channel", "")
message = fields.get("message", "")

# Send
client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])

try:
    result = client.chat_postMessage(channel=channel, text=message)
    response = f"sent: {result['ts']}"
except SlackApiError as e:
    response = f"error: {e.response['error']}"

# Write response
with open("slack-response.txt", "w") as f:
    f.write(response)
