import sounddevice as sd
import argparse
import sys

from input_recording import wait_for_trigger

parser = argparse.ArgumentParser()
parser.add_argument(
    "--is_multi", "-m", help="use multiple speakers", action="store_true"
)
parser.add_argument(
    "--input_device", "-i", help="input sound device\nPlease run bluetooth_sound_device.py.", type=int,
)
parser.add_argument(
    "--output_device", "-o", help="output sound device\nPlease run bluetooth_sound_device.py.", type=int,
)

if __name__ == '__main__':
	print(sd.query_devices())
	args = parser.parse_args()

	if args.input_device:
		input_sound_device = args.input_device
	else:
		input_sound_device = sd.default.device[0]

	if args.output_device:
		output_sound_device = args.output_device
	else:
		output_sound_device = sd.default.device[1]
		
	wait_for_trigger(input_device=input_sound_device)
	sys.exit(0)
