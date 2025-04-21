# An Example Run of the Code for someone based in Seattle

## Code Execution Output

```
Training the keyword detection model...
Spoken output delivered
Recording data/yes_1.wav for 1s…
Recording data/yes_2.wav for 1s…
Recording data/yes_3.wav for 1s…
Recording data/yes_4.wav for 1s…
Recording data/yes_5.wav for 1s…
Recording data/not_yes_1.wav for 1s…
Recording data/not_yes_2.wav for 1s…
Recording data/not_yes_3.wav for 1s…
Recording data/not_yes_4.wav for 1s…
Recording data/not_yes_5.wav for 1s…
Epoch 1/10, Loss: 0.8719
Epoch 2/10, Loss: 0.6660
Epoch 3/10, Loss: 0.6809
Epoch 4/10, Loss: 0.5878
Epoch 5/10, Loss: 0.5301
Epoch 6/10, Loss: 0.4419
Epoch 7/10, Loss: 0.3546
Epoch 8/10, Loss: 0.2526
Epoch 9/10, Loss: 0.2341
Epoch 10/10, Loss: 0.1199
Spoken output delivered
Recognized Speech (offline):  Seattle, Seattle, Seattle, Seattle.
Recorded city: Seattle, Seattle.
Spoken output delivered
Recognized Speech (offline):  I have a casual outfit, 3 formal suits, some winter jackets and a collection of t-shirt and jeans.
Recorded wardrobe:  I have a casual outfit, 3 formal suits, some winter jackets and a collection of t-shirt and jeans.
User information saved to data/user_info.json
Listening for speech input...
Started Listening to User's voice
Recognized Speech (offline):  I would like an outfit recommendation
Recognized Speech (offline):  I would like an outfit recommendation
Spoken output delivered
Recognized Speech (offline):  Yes.
User agreed to take a picture.
Allowing camera to adjust...
User image captured
Detected 1 face(s) for blurring
User image with blurred faces saved as data/user_image_blurred.jpg
Retrieved weather for Seattle, Seattle.: 12.37°C, broken clouds
Opening image for Gemini API...
Sending request to Gemini Vision API...
```

## Outfit Recommendation

Based on your existing wardrobe and the current weather (broken clouds, 12.37°C), here's a detailed outfit recommendation:

**Outfit:**

* **Base:** A comfortable t-shirt and jeans. This is a versatile starting point that can be easily layered.
* **Outer Layer:** Since the temperature is cool, go for one of your winter jackets. This will provide the necessary warmth while remaining suitable for the broken clouds.

**Reasoning:**

* **Temperature:** 12.37°C is cool enough to warrant a jacket, but not so cold that you need heavy layers.
* **Weather:** Broken clouds indicate partly sunny and partly cloudy conditions.
* **Your Wardrobe:** You have a casual outfit, suits, winter jackets, and t-shirts/jeans.

**Additional considerations:**

* **Footwear:** Wear appropriate shoes depending on the weather conditions. If the clouds are threatening rain, opt for water-resistant shoes.

## Audio Output

<audio controls>
    <source src="data/output.mp4" type="audio/mp4">
    Audio Output from Assistant
</audio>

https://github.com/user-attachments/assets/9a583a2f-0c68-4c47-bab2-b6bc9f82d0df


[Listen to the audio](https://gabalpha.github.io/read-audio/?p=https://github.com/likhithayinala/embedded-ai-project/blob/main/data/output.mp3)


