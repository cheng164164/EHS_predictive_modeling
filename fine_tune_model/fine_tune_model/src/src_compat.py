def format_messages_for_eval(messages):
    parts = []
    for m in messages:
        parts.append(f"### {m['role'].upper()}:\n{m['content']}")
    parts.append('### ASSISTANT:\n')
    return '\n\n'.join(parts)
