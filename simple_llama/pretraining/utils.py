import os
import torch


def load_checkpoint(config: any, ckpt_dir: str, ddp: bool, master_process: bool):
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    available_checkpoints = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)]
    if not available_checkpoints:
        raise FileNotFoundError("No checkpoints available in the directory.")

    # Each ckpt path might look like "checkpoints/model_1050M_3110.pth", and sort accordingly
    def parse_model_size(ckpt_name: str):
        if not ckpt_name.endswith(".pth"):  # Invalid, so return 0, moving to end of list
            return 0

        trained_tokens = ckpt_name.split("_")[-3]  # Grabs the part like "150M" or "3B"
        value, unit = int(trained_tokens[:-1]), trained_tokens[-1]

        if unit not in ["M", "B"]:   # Invalid as well, return 0
            return 0

        multiplier = 1 if unit == "M" else 1000
        return value * multiplier


    # Sort checkpoints by tokens trained descending
    available_checkpoints.sort(key=parse_model_size, reverse=True)

    # Print sorted checkpoints
    print("Available checkpoints (sorted by tokens trained):")
    for idx, ckpt_path in enumerate(available_checkpoints):
        print(f"{idx + 1}. {ckpt_path}")
    print("\n")


    if ddp:
        if master_process:
            print(f"User is using DDP. Defaulting to Option 1 (Longest Trained Model)...")
        selected_idx = 0  # Adjust as needed. Can't directly use input() if using DDP when training
    else:
        selected_idx = int(input("Select a checkpoint to load (number): ")) - 1
    path = available_checkpoints[selected_idx]

    print(f"Loading in {path}...")
    ckpt = torch.load(path)

    # Asserting proper hyperparameter alignment
    assert_hyperparameter_equivalence(config, ckpt["config"])

    total_tok_trained = ckpt["total_tok_trained"]

    return ckpt, total_tok_trained


def assert_hyperparameter_equivalence(config1: any, config2: any):
    def _assert_equal(attribute):
        val1 = getattr(config1, attribute)
        val2 = getattr(config2, attribute)

        if val1 != val2:
            raise ValueError(
                f"Mismatch in '{attribute}': {val1} != {val2}"
            )

    _assert_equal("n_embd")
    _assert_equal("n_heads")
    _assert_equal("n_layers")
    _assert_equal("multiple_of")

    if config1.use_mla != config2.use_mla:
        raise ValueError(
            f"Mismatch in 'use_mla': {config1.use_mla} (expected) != {config2.use_mla} (from checkpoint)"
        )

    if config1.use_mla:  # Only check MLA-specific fields if enabled
        _assert_equal("q_lora_rank")
        _assert_equal("kv_lora_rank")
        _assert_equal("qk_nope_head_dim")
        _assert_equal("qk_rope_head_dim")
        _assert_equal("v_head_dim")


def check_log_file_existence(log_file: str, ddp: bool):
    if ddp and os.path.exists(log_file):
        print(f"Log file '{log_file}' already exists.")
        print("DDP usage detected.")
        print("Default Behavior: Deleting log file...")
        os.remove(log_file)
        print("Log file removed")
        return

    if os.path.exists(log_file):
        print(f"Log file '{log_file}' already exists.")
        print("Options:")
        print("1. Delete the existing log file")
        print("2. Use a new log file name")
        print("3. Exit")

        user = input("Enter choice (1/2/3): ").strip()

        if user == "1":
            confirm = input("Confirm deletion of file? [Y/N]: ")
            if confirm.lower() != "y":
                print("Now exiting")
                exit()
            os.remove(log_file)
            print("Deleted original log file.")
        elif user == "2":
            new_name = input("Enter new log file name: ").strip()
            if not new_name:
                print("Invalid filename. Exiting.")
                exit()
            else:
                log_file = new_name + ".txt" if not new_name.endswith(".txt") else new_name
                print(f"Now using new log file: {log_file}")
        else:
            print("Exiting") if user == "3" else print("Invalid response. Now exiting.")
            exit()

        print("\n\n")


def root_path(*subpaths):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return os.path.join(PROJECT_ROOT, *subpaths)


few_shot_prompts = [
    """<SOS>Summarize the short paragraphs below:
Input: The cat saw a mouse darting across the garden and immediately gave chase. It ran past flowerbeds and under fences, determined to catch its prey. After a few minutes, the cat finally cornered the mouse behind a bush and caught it.
Summary: The cat caught the mouse after a chase through the garden.

Input: The company announced a new phone model during a high-profile launch event last month. Initial reviews praised its design and performance. Since release, it has consistently sold out and become the company's most successful product.
Summary: The new phone model became the company's best-seller.

Input: After weeks of drought, heavy rains finally arrived in the region. The parched soil soaked up the water quickly, and farmers were relieved to see their crops revived. Rivers and reservoirs also began to refill, easing concerns about water shortages.
Summary: Rain ended the drought and helped restore crops and water levels.

Input: The artist spent months preparing for the gallery opening, perfecting each piece with careful detail. On opening night, visitors praised the collection, and several paintings were sold within hours. The success marked a major milestone in the artist's career.
Summary: The artist's gallery opening was a major success.

Input: They began their hike early in the morning, carrying supplies and navigating steep trails. As the sun began to set, they finally reached the summit of the mountain. From there, they enjoyed a breathtaking view of the sunset stretching across the horizon.
Summary:""",

    """<SOS>Classify the sentiment as "Positive" or "Negative":
Input: I absolutely loved the movie, it was fantastic!
Sentiment: Positive

Input: The service was terrible and I won't come back.
Sentiment: Negative

Input: This book was boring and hard to finish.
Sentiment: Negative

Input: The app keeps crashing and it's really frustrating.
Sentiment: Negative

Input: I'm impressed by how fast and easy this process was.
Sentiment: Positive

Input: The food was delicious and the atmosphere was wonderful.
Sentiment:""",

    """<SOS>Correct the spelling mistakes:
Input: I relly enjoied the concert last nite.
Corrected: I really enjoyed the concert last night.

Input: She walkked to the storr to bye some bred.
Corrected: She walked to the store to buy some bread.

Input: Theyre going too the bech tommorow.
Corrected: They're going to the beach tomorrow.

Input: He didn't no wich buss to take so he askked someone.
Corrected: He didn't know which bus to take so he asked someone.

Input: Its importent to chek youre answers before submision.
Corrected:""",

    """<SOS>Answer the following math problems:
Input: If you have 5 apples and you buy 3 more, how many apples do you have?
Answer: 8

Input: John had 12 candies and gave 5 to his friend. How many candies does he have now?
Answer: 7

Input: Sarah bought 4 books and then 6 more. How many books does she have?
Answer: 10

Input: A box contains 15 pencils. If 9 pencils are removed, how many are left?
Answer: 6

Input: Emma saved $20 and earned $15 more from chores. How much money does she have now?
Answer:""",

    """<SOS>Write a polite email reply:
Input: Hi, can you send me the report by tomorrow? Thanks.
Reply: Hi, sure thing! I'll send the report to you by tomorrow. Let me know if you need anything else.

Input: Hi, just checking if you're available for a meeting this Friday.
Reply: Hi, thanks for reaching out. I'm available on Friday - what time works best for you?

Input: Hi, could you help me with the project deadline?
Reply: Hi, of course. Let me know what you need help with, and I'll do my best to assist.

Input: Hi, do you have the updated slides for the presentation?
Reply: Hi, yes - I'll send over the updated slides shortly. Let me know if you'd like me to walk you through them.

Input: Hi, do you have time to meet with the manager later today?
Reply:""",

    """<SOS>Rewrite the sentence to sound more professional:
Input: We gotta fix this issue ASAP or the client's gonna be mad.
Professional: We need to resolve this issue promptly to avoid upsetting the client.

Input: I don't think this idea's gonna work out.
Professional: I have concerns about the feasibility of this idea.

Input: Can you send me those docs you talked about?
Professional: Could you please share the documents you mentioned earlier?

Input: This thing keeps breaking and it's super annoying.
Professional: This issue recurs frequently and is causing significant inconvenience.

Input: I told them already, but they didn't get it.
Professional:""",

    """<SOS>Suggest a title for the following:
Input: This article discusses how remote work has changed employee productivity, highlighting both benefits and challenges faced by companies adapting to flexible schedules.
Title: How Remote Work is Reshaping Productivity

Input: A detailed guide on growing tomatoes at home, covering soil preparation, watering techniques, and common pest issues.
Title: A Beginner's Guide to Growing Tomatoes

Input: A blog post about saving money while traveling through Europe by using hostels, discount cards, and local transportation.
Title: Budget Travel Tips for Exploring Europe

Input: This piece explores the impact of social media on teenagers' mental health and self-image, citing recent studies and expert opinions.
Title: The Effects of Social Media on Teen Mental Health

Input: An overview of the latest electric vehicles hitting the market, with comparisons of range, price, and performance.
Title:""",

    """<SOS>Summarize customer reviews in one sentence:
Input: I bought this blender last month, and it's amazing. It crushes ice in seconds and the build quality is fantastic. Cleanup is super easy, and it looks great on my counter. Totally worth the price.
Summary: The blender is powerful, well-built, easy to clean, and worth the price.

Input: The jacket looks great but the zipper broke after a week. The material feels cheap and thin. Customer service was slow to respond. Overall, I regret buying it.
Summary: The jacket has poor quality and bad support, leading to regret.

Input: The headphones are lightweight and very comfortable. The sound is crystal clear with deep bass. Battery life easily lasts over a day of use. Setup was also very simple.
Summary: The headphones are comfortable, have excellent sound, long battery life, and are easy to set up.

Input: This vacuum is so loud it scares my dog. It does pick up dirt well, but the noise is unbearable. Not for pet owners.
Summary: The vacuum is effective but extremely noisy and unsuitable for homes with pets.

Input: Love this desk! Sturdy build, lots of space, and easy to assemble. Looks great in my home office.
Summary:"""

]


