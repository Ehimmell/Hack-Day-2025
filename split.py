import os
import argparse
import soundfile as sf

def split_file(input_path, output_dir):
    # Read audio data
    data, sample_rate = sf.read(input_path, dtype='float32')
    # Convert to mono if stereo
    if data.ndim > 1:
        data = data.mean(axis=1)
    total_samples = len(data)
    half = total_samples // 2

    # Prepare output names
    base = os.path.splitext(os.path.basename(input_path))[0]
    part1_name = f"{base}_part1.wav"
    part2_name = f"{base}_part2.wav"

    os.makedirs(output_dir, exist_ok=True)

    # Write halves
    part1_path = os.path.join(output_dir, part1_name)
    sf.write(part1_path, data[:half], sample_rate)

    part2_path = os.path.join(output_dir, part2_name)
    sf.write(part2_path, data[half:], sample_rate)

    # Delete original
    os.remove(input_path)
    print(f"Split and removed '{input_path}' -> '{part1_name}', '{part2_name}'")


def main(input_dir, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(input_dir, "split")

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.wav'):
            continue
        split_file(os.path.join(input_dir, fname), output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split each WAV file in half using soundfile, save halves, and delete originals"
    )
    parser.add_argument('--input_dir', '-i', required=True,
                        help='Directory containing WAV files to split')
    parser.add_argument('--output_dir', '-o', required=False,
                        help='Directory to save split files (default: input_dir/split)')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
