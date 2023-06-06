"""Template robot with Python."""
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

def minimal_task():
    print("\n---- Starting. ----\n")

    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    sequences = pipeline(
    """You are an assitant that helps with extracting structured information from email threads that talk about payments of the corporate invoices. Your task is to only return JSON data from the thread below.

    --- Thread Starts ---
    TO: john.doe@co.com
    FROM: elin.johnson@someco.com
    SUBJECT: Invoice 1234
    DATETIME: 2023-06-02 12:00:00

    Hi John,
    We are paying the invoice next week monday.
    Br, Elin

    TO: elin.johnson@someco.com
    FROM: john.doe@co.com
    SUBJECT: Invoice 1234
    DATETIME: 2023-06-01 12:00:00

    Hi Elin,

    Your invoice 1234 worth EUR 1,656,545.77 is due next week Tuesday Jun 13th. Could you please confirm if it is going to be paid on time?

    Thanks,
    John

    Accounts Receivable Team Agent
    CompanyCompany Ltd.
    --- Thread Ends ---

    These are the key value pairs to extract from the data in JSON format. Only return the JSON data, do not return any other text such as instructions how to get the JSON with code.

    <invoice_id>: <which invoice is the thread about>,
    <invoice_amount>: <how much is the invoice worth>,
    <invoice_currency>: <what is the currency of the invoice>,
    <confirmation>: <customer's confirmation is the invoice going to be paid on time or not. Only return one of these values: on-time (if customer confirms the payment will happen on time or before), late (if customer says they will pay late), more-info (if customer asks for more information regarding the invoice) or unknown (if there is no information to fill this datapoint)>,
    <customer_name>: <name of the customer>,
    <payment_date>: <when is the payment going to happen, if customer confirms one. If no date is known leave this NULL>

    JSON:
    """,
        max_length=1024,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

    print("\n---- Finished. ----\n")

if __name__ == "__main__":
    minimal_task()
