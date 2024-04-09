from opencompass.models import SkyOpenAI


api_meta_template = dict(
    round=[
            dict(role='HUMAN', api_role='HUMAN'),
            dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(abbr='Sky-GPT4',
        type=SkyOpenAI, path='sky-gpt4',
        meta_template=api_meta_template,
        query_per_second=1,
        batch_size=8),
]
