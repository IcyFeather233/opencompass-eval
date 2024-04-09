import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
import requests
import json
import uuid

from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]

class Sky_GPT_Model:
    def __init__(self, app_key="94edf6d2a98fb1fc930ed44aa60de483", url="http://8.218.90.164:8085/xmind/gpt4turbo"):
        self.url = url
        self.app_key = app_key
        self.headers = {
            "Host": "8.218.90.164:8085",
            "User-Agent": "Go-http-client/1.1",
            "Content-Type": "application/json",
            "App_key": app_key,
            "Accept-Encoding": "gzip"
        }

    def generate_response(self, question):
        timestamp = str(int(time.time() * 1000))

        data = {
            "timeStamp": timestamp,
            "tppBizNo": uuid.uuid4().hex,
            "endUser": "QD-ms",
            "messages": [
                {"role": "user", "content": question}
            ],
            "model": 5,
            "temperature": 0.8,
            "top_p": 1,
            "n": 1,
            "stream": False,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        response = requests.post(self.url, headers=self.headers, data=json.dumps(data))
        res = response.json()
        result_txt = res['completion']['choices'][0]['message']['content']
        return result_txt.strip()


class SkyOpenAI(BaseAPIModel):
    """Skywork内部封装的OpenAI Api

    Args:
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        retry (int): Number of retires if the API call fails. Defaults to 2.
    """
    
    is_api: bool = True

    def __init__(self,
                 path: str = 'sky-gpt4',
                 max_seq_len: int = 2048,
                 query_per_second: int = 1,
                 meta_template: Optional[Dict] = None,
                 retry: int = 5):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         query_per_second=query_per_second,
                         meta_template=meta_template,
                         retry=retry)
        self.sky_gpt_model = Sky_GPT_Model()

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str or PromptList]): A list of strings or PromptDicts.
                The PromptDict should be organized in OpenCompass'
                API format.

        Returns:
            List[str]: A list of generated strings.
        """
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs))
        self.flush()
        return results

    def _generate(
        self,
        input: str or PromptList
    ) -> str:
        """Generate results given an input.

        Args:
            inputs (str or PromptList): A string or PromptDict.
                The PromptDict should be organized in OpenCompass'
                API format.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, (str, PromptList))
        """
        ["USER", "BOT", "USER"]
        必须是基数轮

        """

        if isinstance(input, str):
            messages = [input]
        else:
            messages = []
            for item in input:
                msg = item['prompt']
                messages.append(msg)
        data = messages
        
        # 检查对话必须是基数轮
        if len(data) % 2 != 1:
            raise ValueError("Question Error: data num is not odd: {}".format(str(data)))
        
        # data.update(self.generation_kwargs)

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.acquire()
            try:
                response = self.sky_gpt_model.generate_response(str(data))
            except Exception as err:
                print('Request Error:{}'.format(err))
                time.sleep(1)
                continue

            self.release()

            if response is None:
                print('Connection error, reconnect.')
                # if connect error, frequent requests will casuse
                # continuous unstable network, therefore wait here
                # to slow down the request
                self.wait()
                max_num_retries += 1
                continue
            else:
                return response

        raise RuntimeError(response)
