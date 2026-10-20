# Experimental vision LLM automatic labeling

This branch keeps a vision LLM backend for further evaluation. It is separate
from the Paddle model selection pull request. A valid JSON response does not
guarantee accurate text or polygons; localization quality depends on the chosen
model and service. Local evaluations still showed clipped or misplaced text
boxes. Review every automatic annotation before confirming it.

## Try another vision model

1. Open an image directory, then **Automatic labeling → Automatic labeling
   settings** (Ctrl+M).
2. Choose the **LLM vision model** backend. Enter a vision-capable model name
   and the base URL of an OpenAI-compatible Chat Completions service, such as
   `https://vision.example/v1`.
3. On Windows, enter an API key in the dialog; it is stored using user-bound
   DPAPI encryption. Alternatively, set `PPOCRLABEL_API_KEY` before starting the
   application. Use the environment variable on other platforms. The key is
   not saved in the dataset's `ppocrlabel.json`.
4. Use additional instructions only for dataset-specific targets or
   transcription conventions. The built-in prompt already describes coordinate
   units, complete character boundaries, and the response schema.
5. Start automatic labeling and manually inspect the results. Single-box
   re-recognition continues to use the selected Paddle model.

The request sends the full decoded image as a PNG data URL, its original
dimensions, the built-in annotation prompt, and optional project instructions.
The API key is sent in the authorization header. The service receives the image
and may charge for each request, including an invalid-output correction retry.

The backend expects JSON containing one four-point polygon per text line:

```json
{
  "annotations": [
    {
      "transcription": "TEST123",
      "points": [[20, 30], [140, 30], [140, 60], [20, 60]]
    }
  ]
}
```

Coordinates refer to original-image pixels. Format and coordinate checks run
before results enter the existing pending annotation flow. Confirmed images
are skipped; pending annotations still need human confirmation. Keep
`fileState.txt` with `Label.txt` when moving a dataset.

The interface shows the current image, phase, and elapsed time. Cancellation
discards late results locally; it cannot guarantee cancellation or a refund at
the remote service. One image has a 300-second local deadline shared by the
initial request and an optional correction request.

## Checks and scope

Run `python -m unittest discover -s tests -v`. Protocol tests use generated
images, synthetic credentials, and a local HTTP server. They do not contact a
real model provider and do not measure real model recognition quality.

Coordinate grids, overlapping crop experiments, and visual feedback experiments
have not been integrated into this backend. Private evaluation images, response
captures, local configuration, model weights, and packaged Windows programs
are not included in this branch.
