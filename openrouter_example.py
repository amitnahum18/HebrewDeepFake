"""Run one OpenRouter text request, or explicitly select the speech endpoint."""
import argparse
import json
import sys

from openrouter_client import OpenRouterError, request_speech, request_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Reply with exactly: OpenRouter connection successful.")
    parser.add_argument("--model", help="Exact provider/model slug; text defaults to OPENROUTER_MODEL or openai/gpt-4o-mini.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--json", action="store_true", help="Print request/usage metadata with the result.")
    parser.add_argument("--speech", action="store_true", help="Generate speech instead of a chat response.")
    parser.add_argument("--voice", help="Voice ID from the selected speech model's catalog.")
    parser.add_argument("--format", choices=["mp3", "pcm"], default="mp3")
    parser.add_argument("--output", help="New output filename ending in .mp3 or .pcm.")
    args = parser.parse_args()
    if args.speech and not all([args.model, args.voice, args.output]):
        parser.error("--speech requires --model, --voice and --output")
    try:
        if args.speech:
            result = request_speech(args.prompt, model=args.model, voice=args.voice, output=args.output,
                                    response_format=args.format, timeout=args.timeout)
        else:
            result = request_text(args.prompt, model=args.model, max_tokens=args.max_tokens, timeout=args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json or args.speech else result["text"])
        return 0
    except OpenRouterError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("Error: Could not read or write a local output file; check the path and permissions.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
