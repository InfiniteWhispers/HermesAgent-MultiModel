#!/usr/bin/env python3
"""
Local Mixture-of-Agents helper.  Does NOT rely on the Hermes library.
Queries reference models in parallel via the Ollama HTTP API, then asks
an aggregator model to synthesize the results.

Invoke via Hermes with:
  /localmoa <question> [flags]
  @localmoa <question> [flags]

Or directly:
  ~/.hermes/bin/local_moa.py --prompt "Your question" [flags]
"""

import argparse, json, os, sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from json import JSONDecodeError

def _normalize_ollama_url(raw: str) -> str:
    """Accept a bare host, a /v1 base, or a full /api/chat endpoint.

    Hermes sets OLLAMA_BASE_URL to a base host (http://localhost:11434), but this
    script needs the native chat endpoint. Normalize so it works either way.
    """
    raw = (raw or "").rstrip("/")
    if raw.endswith("/api/chat"):
        return raw
    if raw.endswith("/v1"):            # OpenAI-compat base → native chat path
        raw = raw[:-3]
    return raw + "/api/chat"


DEFAULT_OLLAMA_URL  = _normalize_ollama_url(os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
DEFAULT_REFERENCES  = ["qwen3:14b", "ornith-9b"]
DEFAULT_AGGREGATOR  = "gpt-oss-20b"
DEFAULT_NUM_CTX     = 65536
DEFAULT_TIMEOUT     = 200  # seconds per reference request


def ollama_prompt(
    model: str,
    prompt: str,
    *,
    system: str = "",
    temperature: float | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    timeout: int = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options: dict = {"num_ctx": num_ctx}
    if temperature is not None:
        options["temperature"] = temperature

    data = {"model": model, "messages": messages, "stream": False, "options": options}
    req = Request(
        base_url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))["message"]["content"]
    except TimeoutError:
        raise RuntimeError(f"Model '{model}' timed out after {timeout}s") from None
    except HTTPError as e:
        if e.code == 404:
            raise RuntimeError(f"Model '{model}' not found in Ollama (is it pulled?)") from e
        raise RuntimeError(f"Ollama returned HTTP {e.code} for '{model}': {e}") from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach Ollama ({base_url}): {e}") from e
    except (KeyError, JSONDecodeError) as e:
        raise RuntimeError(f"Unexpected response from '{model}': {e}") from e


def moa(
    prompt: str,
    references: list[str],
    aggregator: str,
    *,
    system: str = "",
    temperature: float | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    timeout: int = DEFAULT_TIMEOUT,
    base_url: str = DEFAULT_OLLAMA_URL,
    verbose: bool = False,
) -> str:
    def _query(ref: str) -> tuple[str, str]:
        return ref, ollama_prompt(
            ref, prompt,
            system=system, temperature=temperature,
            num_ctx=num_ctx, timeout=timeout, base_url=base_url,
        )

    results: dict[str, str] = {}
    errors:  dict[str, str] = {}

    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(_query, ref): ref for ref in references}
        for future in as_completed(futures):
            ref = futures[future]
            try:
                _, out = future.result()
                results[ref] = out
                if verbose:
                    print(f"\n=== {ref} ===\n{out}", file=sys.stderr)
            except Exception as e:
                errors[ref] = str(e)
                print(f"[WARNING] {ref} failed: {e}", file=sys.stderr)

    if not results:
        detail = "\n".join(f"  {m}: {e}" for m, e in errors.items())
        raise RuntimeError(f"All reference models failed:\n{detail}")

    # Anonymize model labels to prevent aggregator bias
    ordered = [ref for ref in references if ref in results]
    labeled = {f"Model {i + 1}": results[ref] for i, ref in enumerate(ordered)}

    agg_instruction = (
        f"You have responses from {len(labeled)} models to the following question:\n"
        f"  {prompt}\n\n"
        f"Model responses:\n{json.dumps(labeled, indent=2)}\n\n"
        "Synthesize a single definitive answer. Extract the strongest and most accurate "
        "points from each response. Where models disagree, apply critical judgment to "
        "resolve the contradiction. The final answer should be more complete and accurate "
        "than any individual response. Stay focused on the original question."
    )
    # Aggregator gets extra time — synthesis of N responses is heavier than a reference call
    return ollama_prompt(
        aggregator, agg_instruction,
        num_ctx=num_ctx, timeout=timeout + 60, base_url=base_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Local Mixture-of-Agents: query multiple Ollama models in parallel then synthesize.\n\n"
            "Invoke via Hermes:\n"
            "  /localmoa <question> [flags]\n"
            "  @localmoa <question> [flags]\n\n"
            "Or directly:\n"
            "  ~/.hermes/bin/local_moa.py --prompt \"Your question\" [flags]"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Model tags must be Ollama pull tags (e.g. qwen3:14b, gpt-oss-20b),\n"
            "not Hermes config aliases.\n\n"
            "Alias → Ollama tag mapping:\n"
            "  qwen3-14b-think    → qwen3:14b\n"
            "  qwythos-9b         → qwythos-9b\n"
            "  gpt-oss-20b        → gpt-oss-20b\n"
            "  ornith-9b          → ornith-9b\n"
            "  gemma4-heretic-12b → gemma4-heretic-12b\n"
            "  qwen25-coder-7b    → qwen2.5-coder:7b\n"
            "  qwen3-vl-8b        → qwen3-vl:8b\n\n"
            "OLLAMA_BASE_URL env var overrides the default Ollama endpoint."
        ),
    )
    parser.add_argument(
        "--prompt", required=True,
        help="The question or task sent to all reference models"
    )
    parser.add_argument(
        "--references", nargs="+", default=DEFAULT_REFERENCES,
        help=f"Reference model tags queried in parallel (default: {' '.join(DEFAULT_REFERENCES)})"
    )
    parser.add_argument(
        "--aggregator", default=DEFAULT_AGGREGATOR,
        help=f"Aggregator model tag that synthesizes the final answer (default: {DEFAULT_AGGREGATOR})"
    )
    parser.add_argument(
        "--system", default="",
        help="System prompt injected into all reference model calls (not the aggregator)"
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature — 0.2-0.4 for factual, unset for general/creative (default: model default)"
    )
    parser.add_argument(
        "--num-ctx", type=int, default=DEFAULT_NUM_CTX,
        help=f"Context window tokens passed to Ollama (default: {DEFAULT_NUM_CTX})"
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Per-request socket timeout in seconds (default: {DEFAULT_TIMEOUT}; aggregator gets +60s). "
             "Use 300 when running 3+ cold models."
    )
    parser.add_argument(
        "--ollama-url", default=DEFAULT_OLLAMA_URL,
        help="Ollama chat endpoint (default: $OLLAMA_BASE_URL or http://localhost:11434/api/chat)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print each reference model's raw response to stderr before aggregating"
    )
    args = parser.parse_args()

    output = moa(
        args.prompt, args.references, args.aggregator,
        system=args.system,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        timeout=args.timeout,
        base_url=_normalize_ollama_url(args.ollama_url),
        verbose=args.verbose,
    )
    print(output)


if __name__ == "__main__":
    main()
