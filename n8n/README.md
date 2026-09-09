# Optional n8n downstream notifications

This directory contains a **workflow export only** — n8n itself is not installed, configured, or
run as part of this repository, and no credentials of any kind are included or invented.

## What this is

`triage-notifications.workflow.json` is a minimal n8n workflow that:

1. Receives a webhook POST for every newly finalized **successful** triage result (not filtered
   by escalation or priority — every result is sent).
2. Branches deterministically on the already-decided fields:
   - `body.escalated == true` → escalation notification path (checked **first**, so an
     escalated-and-high-priority result triggers exactly one notification, not two).
   - otherwise, `body.priority == "high"` → urgent high-priority notification path.
   - otherwise → no action.
3. The two notification branches end in **Gmail "Send" nodes** with no recipient and no
   credentials configured — they build the email subject/body from the payload and stop there.
   You attach your own Gmail OAuth2 credentials and set the recipient address manually in the n8n
   UI (see below) before either node can actually send anything.
   - Escalation branch — subject `Human Review Required - Smart Inquiry Triage`.
   - Urgent high-priority branch — subject `Urgent High-Priority Inquiry - Smart Inquiry Triage`.
   - Both bodies include the customer inquiry, predicted category, priority, assigned queue,
     confidence, and the AI-generated resolution note, plus a plain-language statement of why the
     email was sent.

**These two emails are internal operational notifications to whoever is staffing the relevant
queue — they are not, and must never become, a customer-facing automated email path.** Nothing in
this workflow emails the customer who submitted the inquiry; the recipient is always an internal
operator/reviewer mailbox. In this demo, a personal Gmail address stands in for that operational
mailbox purely for convenience — in a real deployment the recipient would be a team distribution
list or a ticketing-system inbox, not an individual's personal account.

n8n makes no AI decisions. It only receives the already-finalized result from
`src/notifications.py` and applies simple, independently-changeable business rules to it.

## Manual setup (not done for you)

1. Install/run n8n yourself (e.g. `npx n8n`, Docker, or n8n Cloud) — not part of this repo.
2. In the n8n editor, **Import from File** and select `triage-notifications.workflow.json`.
3. Open the "Triage Result Webhook" node. For manual testing, copy its **Test URL**
   and select **Listen for Test Event** before submitting an inquiry. For ongoing use,
   copy its **Production URL** after the workflow is published/active.
4. Set that URL as the `TRIAGE_N8N_WEBHOOK_URL` environment variable wherever you run
   `streamlit run app/app.py`. Leaving it unset keeps the integration fully disabled.
5. Open each Gmail node ("...configure manually") and:
   - Attach your own Gmail OAuth2 credentials via n8n's Credentials panel (create/select them in
     the node's Credential dropdown — n8n handles the OAuth flow itself, nothing to paste here).
   - Set the `sendTo` field to whichever recipient address should receive that notification. It
     ships empty; the node cannot send until you fill this in.
6. Publish/activate the workflow and use its **Production URL** for ongoing results.
   Activation alone does not enable the Test URL; testing requires the listener in step 3.

## Payload received by the webhook

Exactly the fields already produced by `src/main.py`'s `triage_inquiry()`, unrenamed:

```json
{
  "query": "...",
  "category": "...",
  "priority": "low | medium | high",
  "routed_queue": "...",
  "confidence": 0.0,
  "escalated": true,
  "resolution_notes": "..."
}
```

`retrieved_past_cases` is intentionally not sent — it isn't needed for the notification routing
rules above and keeping the payload minimal keeps this integration's one job clear.

**Important — n8n's Webhook node wraps this JSON body under a top-level `body` key.** Every
expression in this workflow that reads a triage-result field therefore uses
`{{$json["body"]["<field>"]}}` (e.g. `{{$json["body"]["escalated"]}}`), never the bare
`{{$json["<field>"]}}`. This was verified against a real Streamlit/LangGraph-generated payload —
an earlier version of this workflow used the bare form and every condition silently evaluated to
`undefined`. If you add new nodes that reference the payload, remember to read from `body` too.

## Failure behavior

The application makes a synchronous webhook POST after successful triage with a 3-second
HTTP timeout. If n8n is unreachable, slow, or returns an error, `src/notifications.py` catches and logs it —
the Streamlit triage result already shown to the user is unaffected either way. There is no
retry.
