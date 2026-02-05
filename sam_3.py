import argparse
import numpy as np
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def apply_mask_overlay(frame_bgr, mask, color=(0, 255, 0), alpha=0.5):
	overlay = frame_bgr.copy()
	overlay[mask] = color
	return cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0)


def pick_best_mask(masks, scores, target_hw):
	if len(scores) == 0:
		return None
	best_idx = int(np.argmax(np.asarray(scores)))
	mask = np.asarray(masks[best_idx])
	if mask.ndim == 3:
		mask = mask[0]
	mask = mask > 0
	if mask.shape != target_hw:
		mask = cv2.resize(mask.astype(np.uint8), target_hw[::-1], interpolation=cv2.INTER_NEAREST)
		mask = mask > 0
	return mask


def run_image(processor, image_path, prompt):
	image = Image.open(image_path)
	state = processor.set_image(image)
	state = processor.set_text_prompt(prompt, state)
	masks = state["masks"]
	scores = state["scores"]

	if len(scores) == 0:
		raise SystemExit("No masks found")

	best_idx = int(np.argmax(np.asarray(scores)))
	mask = np.asarray(masks[best_idx])
	if mask.ndim == 3:
		mask = mask[0]
	mask = (mask > 0).astype(np.uint8) * 255

	masked = image.convert("RGBA")
	masked.putalpha(Image.fromarray(mask))
	masked.save("masked.png")
	masked.show()


def run_camera(processor, camera_index, prompt):
	cap = cv2.VideoCapture(camera_index)
	if not cap.isOpened():
		raise SystemExit(f"Cannot open camera index {camera_index}")

	while True:
		ok, frame = cap.read()
		if not ok:
			break

		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(rgb)
		state = processor.set_image(image)
		state = processor.set_text_prompt(prompt, state)
		mask = pick_best_mask(state["masks"], state["scores"], frame.shape[:2])

		if mask is not None:
			frame = apply_mask_overlay(frame, mask)

		cv2.imshow("SAM3 Camera", frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()


def main():
	parser = argparse.ArgumentParser(description="SAM3 image or camera segmentation")
	parser.add_argument("--prompt", default="person", help="Text prompt for segmentation")
	parser.add_argument("--image", default="../photo.png", help="Path to an image file")
	parser.add_argument("--camera", type=int, help="Camera index for live segmentation")
	args = parser.parse_args()

	# Load model (auto-downloads weights on first run)
	model = build_sam3_image_model()
	processor = Sam3Processor(model, confidence_threshold=0.5)

	if args.camera is not None:
		run_camera(processor, args.camera, args.prompt)
	else:
		run_image(processor, args.image, args.prompt)


if __name__ == "__main__":
	try:
		import cv2
	except Exception as exc:
		raise SystemExit(
			"OpenCV is required for camera mode. Install with: pip install opencv-python"
		) from exc
	main()