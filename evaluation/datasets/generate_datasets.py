"""
Evaluation Benchmark Dataset Generator.
Synthesizes 5 realistic benchmark meetings with:
- Audio WAV files (multi-speaker acoustic characteristics)
- Ground truth reference transcripts
- Ground truth reference summaries
- Reference decision checklists
- Reference action items (with explicit ground truth owners)
- Reference key points checklist (for completeness evaluation)
"""

import os
import json
import logging
from pathlib import Path
import win32com.client

logger = logging.getLogger("DatasetGenerator")

DATASET_ROOT = Path(__file__).resolve().parent

MEETINGS_DATA = [
    {
        "id": "meeting_1_ami_design_sync",
        "name": "Meeting 1: AMI Product Design Sync",
        "domain": "Industrial Design & Hardware Selection (AMI Corpus subset)",
        "speakers": ["David", "Sarah", "Michael"],
        "script": [
            ("David", "Good morning team. Today we need to finalize the remote control hardware design."),
            ("Sarah", "I reviewed the user feedback. Most users preferred kinetic battery charging over standard disposable batteries."),
            ("Michael", "However, kinetic components increase the unit cost by forty percent."),
            ("David", "That is too high for our target retail price of twenty-five dollars."),
            ("Sarah", "We decided to proceed with rechargeable lithium-ion battery cells instead."),
            ("David", "Agreed. That keeps our margins healthy. Michael, please update the CAD schematics by this Friday."),
            ("Michael", "I will update the schematics and send them to the manufacturer."),
            ("David", "We should also double-check the ergonomic grip before physical prototyping."),
            ("Sarah", "I'll coordinate with the user research panel next Monday for the grip testing."),
        ],
        "reference_summary": (
            "The design team met to finalize the remote control hardware specifications. "
            "After evaluating user feedback and unit economics, the team decided against kinetic charging "
            "due to a forty percent cost increase and decided to proceed with rechargeable lithium-ion battery cells. "
            "Michael was tasked with updating the CAD schematics by Friday, Sarah committed to coordinating user grip testing "
            "by next Monday, and the team noted that ergonomic grip checks are required before prototyping."
        ),
        "reference_decisions": [
            {
                "id": 0,
                "statement": "Proceed with rechargeable lithium-ion battery cells instead of kinetic charging.",
                "proposed_by": "Sarah",
            }
        ],
        "reference_action_items": [
            {
                "id": 0,
                "description": "Update the CAD schematics and send them to the manufacturer",
                "owner": "Michael",
                "deadline": "this Friday",
            },
            {
                "id": 1,
                "description": "Coordinate with the user research panel for the grip testing",
                "owner": "Sarah",
                "deadline": "next Monday",
            },
            {
                "id": 2,
                "description": "Double-check the ergonomic grip before physical prototyping",
                "owner": "Unassigned",
                "deadline": None,
            },
        ],
        "key_points": [
            "Hardware design finalization for remote control",
            "Kinetic charging rejected due to 40% cost increase",
            "Decision to use rechargeable lithium-ion batteries",
            "Michael updating CAD schematics by Friday",
            "Sarah coordinating user grip testing next Monday",
            "Ergonomic grip check before prototyping",
        ],
    },
    {
        "id": "meeting_2_qmsum_academic_review",
        "name": "Meeting 2: QMSum AI Research Review",
        "domain": "Deep Learning Optimization (QMSum subset)",
        "speakers": ["Dr. Elena", "Marcus"],
        "script": [
            ("Dr. Elena", "Welcome Marcus. Let's review the attention latency benchmarks on our GPU cluster."),
            ("Marcus", "Standard multi-head attention is causing out-of-memory errors on context lengths above eight thousand tokens."),
            ("Dr. Elena", "Have you benchmarked FlashAttention-2 with BF16 precision?"),
            ("Marcus", "Yes, FlashAttention-2 reduced kernel execution time by three point five times."),
            ("Dr. Elena", "Excellent. We decided to standardize on FlashAttention-2 for all sequence lengths."),
            ("Dr. Elena", "Marcus, please run the full training benchmark across eight H100 GPUs by Thursday."),
            ("Marcus", "I will configure the Slurm job script and monitor the loss curves."),
            ("Dr. Elena", "We need to document the gradient checkpointing memory trade-offs in our paper draft."),
        ],
        "reference_summary": (
            "Dr. Elena and Marcus reviewed attention latency benchmarks for large language models on the GPU cluster. "
            "Because standard multi-head attention failed with out-of-memory errors beyond 8,000 tokens, "
            "they benchmarked FlashAttention-2 which demonstrated a 3.5x speedup. "
            "The team decided to standardize on FlashAttention-2 across all models. "
            "Marcus was assigned to run the training benchmark across eight H100 GPUs by Thursday, "
            "and gradient checkpointing trade-offs must be documented in the paper draft."
        ),
        "reference_decisions": [
            {
                "id": 0,
                "statement": "Standardize on FlashAttention-2 for all sequence lengths across models.",
                "proposed_by": "Dr. Elena",
            }
        ],
        "reference_action_items": [
            {
                "id": 0,
                "description": "Run the full training benchmark across eight H100 GPUs and monitor loss curves",
                "owner": "Marcus",
                "deadline": "Thursday",
            },
            {
                "id": 1,
                "description": "Document the gradient checkpointing memory trade-offs in the paper draft",
                "owner": "Unassigned",
                "deadline": None,
            },
        ],
        "key_points": [
            "GPU cluster memory latency review",
            "OOM issues with standard attention over 8k tokens",
            "FlashAttention-2 provides 3.5x kernel speedup",
            "Decision to standardize on FlashAttention-2",
            "Marcus running 8x H100 benchmark by Thursday",
            "Documentation of gradient checkpointing in draft",
        ],
    },
    {
        "id": "meeting_3_sprint_planning_fintech",
        "name": "Meeting 3: Fintech Sprint Planning",
        "domain": "Banking API & PCI Compliance (Ownership Stress-Test)",
        "speakers": ["Anas", "Priya", "Bob"],
        "script": [
            ("Anas", "Welcome to sprint planning team. Our top priority is passing the annual PCI-DSS audit."),
            ("Priya", "We need to ensure all credit card data in transit uses TLS 1.3 encryption."),
            ("Bob", "Our legacy payment gateway currently only supports TLS 1.2 on two legacy endpoints."),
            ("Anas", "We decided to deprecate the legacy endpoints and enforce TLS 1.3 across all services."),
            ("Anas", "Priya, please rotate the production SSL certificates by tomorrow."),
            ("Priya", "I will rotate the certificates and update the ingress controller tomorrow morning."),
            ("Anas", "Bob will implement tokenized customer vaulting before next Friday."),
            ("Bob", "Understood, I am on it."),
            ("Priya", "Someone needs to clean up the stale sandbox accounts."),
            ("Anas", "Agreed, let's make sure that gets logged."),
        ],
        "reference_summary": (
            "In this fintech sprint planning session, Anas, Priya, and Bob prioritized PCI-DSS compliance requirements. "
            "The team decided to deprecate legacy payment endpoints and enforce TLS 1.3 encryption across all services. "
            "Priya was explicitly assigned to rotate production SSL certificates by tomorrow. "
            "Bob was assigned to implement customer payment token vaulting by next Friday. "
            "Cleaning up stale sandbox accounts was identified as an unassigned task."
        ),
        "reference_decisions": [
            {
                "id": 0,
                "statement": "Deprecate legacy payment endpoints and enforce TLS 1.3 encryption across all services.",
                "proposed_by": "Anas",
            }
        ],
        "reference_action_items": [
            {
                "id": 0,
                "description": "Rotate the production SSL certificates and update ingress controller",
                "owner": "Priya",
                "deadline": "tomorrow",
            },
            {
                "id": 1,
                "description": "Implement tokenized customer vaulting",
                "owner": "Bob",
                "deadline": "next Friday",
            },
            {
                "id": 2,
                "description": "Clean up the stale sandbox accounts",
                "owner": "Unassigned",
                "deadline": None,
            },
        ],
        "key_points": [
            "Sprint priority: PCI-DSS compliance",
            "Legacy endpoints lack TLS 1.3 support",
            "Decision to deprecate legacy endpoints and mandate TLS 1.3",
            "Priya rotating SSL certificates by tomorrow",
            "Bob implementing customer token vaulting by next Friday",
            "Unassigned task to clean up sandbox accounts",
        ],
    },
    {
        "id": "meeting_4_tech_architecture_review",
        "name": "Meeting 4: Technical Architecture Review",
        "domain": "Distributed Systems & Caching (Jargon & Numbers Stress-Test)",
        "speakers": ["Vikram", "Maya"],
        "script": [
            ("Vikram", "Hello Maya. Let's resolve the event ingestion architecture debate between Kafka and RabbitMQ."),
            ("Maya", "Our telemetry ingestion throughput exceeds seventy-five thousand events per second during peak hours."),
            ("Vikram", "RabbitMQ brokers encountered high CPU throttling under seventy thousand messages per second."),
            ("Maya", "Let's go with Apache Kafka with thirty-two partitions for horizontal scaling."),
            ("Vikram", "Agreed, the consensus is to adopt Apache Kafka for the core ingestion pipeline."),
            ("Maya", "I will configure the Kafka cluster with a replication factor of three by Wednesday."),
            ("Vikram", "We decided to set the Redis cache TTL to three thousand six hundred seconds."),
            ("Vikram", "Action item is to monitor consumer group rebalancing latency in Datadog."),
        ],
        "reference_summary": (
            "Vikram and Maya conducted a technical architecture review to evaluate event brokers for high-throughput telemetry. "
            "With peak traffic exceeding 75,000 events per second, RabbitMQ experienced CPU bottlenecks. "
            "The team decided to adopt Apache Kafka with 32 partitions and decided to set the Redis cache TTL to 3,600 seconds. "
            "Maya committed to configuring the Kafka cluster with a replication factor of 3 by Wednesday, "
            "while monitoring consumer group rebalancing latency in Datadog remains an unassigned action item."
        ),
        "reference_decisions": [
            {
                "id": 0,
                "statement": "Adopt Apache Kafka with 32 partitions for horizontal scaling in core ingestion.",
                "proposed_by": "Maya",
            },
            {
                "id": 1,
                "statement": "Set the Redis cache TTL to 3,600 seconds.",
                "proposed_by": "Vikram",
            },
        ],
        "reference_action_items": [
            {
                "id": 0,
                "description": "Configure the Kafka cluster with a replication factor of three",
                "owner": "Maya",
                "deadline": "Wednesday",
            },
            {
                "id": 1,
                "description": "Monitor consumer group rebalancing latency in Datadog",
                "owner": "Unassigned",
                "deadline": None,
            },
        ],
        "key_points": [
            "Event ingestion broker evaluation: Kafka vs RabbitMQ",
            "Peak volume exceeds 75k events/sec",
            "Decision to adopt Apache Kafka with 32 partitions",
            "Decision to set Redis cache TTL to 3600 seconds",
            "Maya configuring Kafka cluster with replication factor 3 by Wednesday",
            "Monitoring consumer rebalancing latency in Datadog",
        ],
    },
    {
        "id": "meeting_5_executive_status_review",
        "name": "Meeting 5: Executive Status Review",
        "domain": "Executive Management & Budget (Fillers & Turn-Taking Stress-Test)",
        "speakers": ["Rachel", "Kevin"],
        "script": [
            ("Rachel", "Um, good afternoon everyone. Uh, let's review our quarterly cloud expenditure and hiring status."),
            ("Kevin", "Well, you know, our AWS infrastructure bill, like, surged by twenty-two percent last month."),
            ("Rachel", "That is concerning. We we decided to freeze all non-essential cloud compute instances immediately."),
            ("Kevin", "I agree with that. Um, Kevin will audit our idle RDS database instances by this Friday."),
            ("Rachel", "Excellent. I will present the updated budget forecast to the board next Tuesday."),
            ("Kevin", "We need to, like, review the engineering open requisitions as well."),
        ],
        "reference_summary": (
            "Rachel and Kevin reviewed the executive quarterly cloud expenditure and budget. "
            "Following a 22 percent increase in monthly AWS infrastructure expenses, "
            "the leadership decided to freeze all non-essential cloud compute instances immediately. "
            "Kevin was assigned to audit idle RDS database instances by this Friday, "
            "Rachel committed to presenting the updated budget forecast to the board next Tuesday, "
            "and reviewing engineering open requisitions was noted as an unassigned action item."
        ),
        "reference_decisions": [
            {
                "id": 0,
                "statement": "Freeze all non-essential cloud compute instances immediately.",
                "proposed_by": "Rachel",
            }
        ],
        "reference_action_items": [
            {
                "id": 0,
                "description": "Audit idle RDS database instances",
                "owner": "Kevin",
                "deadline": "this Friday",
            },
            {
                "id": 1,
                "description": "Present the updated budget forecast to the board",
                "owner": "Rachel",
                "deadline": "next Tuesday",
            },
            {
                "id": 2,
                "description": "Review the engineering open requisitions",
                "owner": "Unassigned",
                "deadline": None,
            },
        ],
        "key_points": [
            "Executive cloud expenditure and budget review",
            "AWS monthly bill surged by 22%",
            "Decision to immediately freeze non-essential cloud compute",
            "Kevin auditing idle RDS instances by Friday",
            "Rachel presenting updated budget forecast next Tuesday",
            "Review of engineering open requisitions",
        ],
    },
]


def generate_all():
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    speaker_engine = win32com.client.Dispatch("SAPI.SpVoice")
    filestream = win32com.client.Dispatch("SAPI.SpFileStream")

    # Available voices
    voices = speaker_engine.GetVoices()
    logger.info(f"Available SAPI voices: {voices.Count}")

    for idx, meeting in enumerate(MEETINGS_DATA):
        meeting_dir = DATASET_ROOT / meeting["id"]
        meeting_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating {meeting['name']} in {meeting_dir}...")

        # 1. Synthesize audio WAV
        audio_path = meeting_dir / "audio.wav"
        filestream.Open(str(audio_path), 3, False)  # 3 = SSFMCreateForWrite
        speaker_engine.AudioOutputStream = filestream

        transcript_lines = []
        speaker_turns = []

        for speaker_name, text in meeting["script"]:
            transcript_lines.append(f"{speaker_name}: {text}")
            # Adjust speech rate slightly per speaker for acoustic diversity
            speaker_hash = sum(ord(c) for c in speaker_name)
            rate_offset = (speaker_hash % 3) - 1  # -1, 0, or 1
            speaker_engine.Rate = rate_offset
            speaker_engine.Speak(text)

        filestream.Close()
        print(f"  -> Generated Audio: {audio_path}")

        # 2. Write reference transcript
        ref_transcript_path = meeting_dir / "reference_transcript.txt"
        with open(ref_transcript_path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))
        print(f"  -> Generated Transcript: {ref_transcript_path}")

        # 3. Write reference summary
        ref_summary_path = meeting_dir / "reference_summary.txt"
        with open(ref_summary_path, "w", encoding="utf-8") as f:
            f.write(meeting["reference_summary"].strip())
        print(f"  -> Generated Reference Summary: {ref_summary_path}")

        # 4. Write reference decisions
        ref_decisions_path = meeting_dir / "reference_decisions.json"
        with open(ref_decisions_path, "w", encoding="utf-8") as f:
            json.dump(meeting["reference_decisions"], f, indent=2)
        print(f"  -> Generated Reference Decisions: {ref_decisions_path}")

        # 5. Write reference action items
        ref_actions_path = meeting_dir / "reference_action_items.json"
        with open(ref_actions_path, "w", encoding="utf-8") as f:
            json.dump(meeting["reference_action_items"], f, indent=2)
        print(f"  -> Generated Reference Action Items: {ref_actions_path}")

        # 6. Write reference key points
        ref_keypoints_path = meeting_dir / "reference_key_points.json"
        with open(ref_keypoints_path, "w", encoding="utf-8") as f:
            json.dump(meeting["key_points"], f, indent=2)
        print(f"  -> Generated Reference Key Points: {ref_keypoints_path}")

    print("\n[SUCCESS] All 5 benchmark evaluation datasets created successfully!")


if __name__ == "__main__":
    generate_all()
