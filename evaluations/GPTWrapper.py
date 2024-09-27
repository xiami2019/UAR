from threading import Event, Thread
import backoff
import json
import os
import datetime
from time import sleep
from watchdog.observers import Observer
from typing import List, Callable, Iterable, Dict, Union, Tuple
from watchdog.events import FileSystemEventHandler, FileModifiedEvent
from multiprocessing import Process, Manager, Queue
import inspect
import queue
import openai


def _wrapper_func(pid, result_queue: Queue, error_queue: Queue, wrapper: 'GPTWrapper', func: Callable, data_chunk, *args, **kwargs):
    try:
        result = func(pid, wrapper, data_chunk, *args, **kwargs)
        result_queue.put(result)
    except Exception as e:
        error_queue.put((pid, e))

def _generate_response(wrapper: 'GPTWrapper', engine: str, messages: List[Union[Dict, str]], fout: str, **kwargs):
    results = []
    for message in messages:
        get_tokens = kwargs.pop('get_tokens', False)
        result = {}
        if get_tokens:
            response, input_tokens, output_tokens = wrapper.completions_with_backoff(
                messages=message,
                engine=engine,
                get_tokens=get_tokens,
                **kwargs
            )
            result['input_tokens'] = input_tokens
            result['output_tokens'] = output_tokens
        else:
            response = wrapper.completions_with_backoff(
                messages=message,
                engine=engine,
                get_tokens=get_tokens,
                **kwargs
            )
        system_prompt = message[0]['content'] if type(message) == list and len(message) == 2 else None
        prompt = message[1]['content'] if type(message) == list and len(message) == 2 else message
        result = {
            'system_prompt': system_prompt,
            'prompt': prompt,
            'response': response
        }
        results.append(result)
        with open(fout, 'a', encoding='utf-8') as fp:
            fp.write(json.dumps(result, ensure_ascii=False)+'\n')
        
    return results

class CustomHandler(FileSystemEventHandler):
    def __init__(self, event):
        self.event_to_set = event

    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent) and event.src_path.endswith('config.json'):
            with open(event.src_path, 'r') as f:
                try:
                    config = json.load(f)
                except Exception as e:
                    print(f'Error: {e}\ncontinue waiting...')
                    return
            self.event_to_set.set()


class GPTWrapper:
    def __init__(self, config_path, base_wait_time=30, lark_hook=None, bias=0) -> None:
        self.config_path = config_path
        self.bias = bias
        config = json.load(open(self.config_path, 'r', encoding='utf-8'))
        self.key_index = config['key_index']
        self.key_list = config['key_list']
        # add a bias to key_list to support multi thread processing
        self.key_list = [self.key_list[(i - self.bias) % len(self.key_list)] for i in range(len(self.key_list))]
        try:
            self.lark_bot = LarkBot(lark_hook)
        except Exception as e:
            print(f'Error: {e}\nLark notice is not available.')
            print(f'Will run without Lark notice.')
            self.lark_bot = None
        self.base_wait_time = base_wait_time
        self.client = openai.OpenAI(
            api_key=self.key_list[self.key_index].get('api_key', None),
            organization=self.key_list[self.key_index].get('organization', None),
            base_url=self.key_list[self.key_index].get('base_url', None)
            )
        
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'lark_bot' in state:
            del state['lark_bot']
        if 'client' in state:
            state['client'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'lark_hook' in state:
            try:
                self.lark_bot = LarkBot(state['lark_hook'])
            except Exception as e:
                print(f'Error: {e}\nLark notice is not available.')
                print(f'Will run without Lark notice.')
                self.lark_bot = None
        
        if 'key_list' in state and 'key_index' in state:
            # 假设需要基于key_list和key_index重建client
            self.client = openai.OpenAI(
                api_key=self.key_list[self.key_index].get('api_key', None),
                organization=self.key_list[self.key_index].get('organization', None),
                base_url=self.key_list[self.key_index].get('base_url', None)
            )

    def __send_message_periodically(self, stop_event):
        wait_turn = 0
        while not stop_event.is_set():
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.lark_bot.send(f'OpenAI: You exceeded ALL your current quota. Please update `config.json` file to resume.\nTimeStamp: {current_time}')
            sleep(2**wait_turn*self.base_wait_time)
            wait_turn += 1
        
    def set_api_key(self):
        self.key_index += 1
        while self.key_index >= len(self.key_list):
            config = json.load(open(self.config_path, 'r', encoding='utf-8'))
            key_list = config['key_list']
            key_index = config['key_index']
            if self.key_list != key_list:
                self.key_list = key_list
                self.key_list = [self.key_list[(i - self.bias) % len(self.key_list)] for i in range(len(self.key_list))]
                self.key_index = key_index
            else:
                event = Event()
                event_handler = CustomHandler(event)
                observer = Observer()
                observer.schedule(event_handler, os.path.dirname(self.config_path), recursive=False)
                observer.start()
                
                if self.lark_bot:
                    message_thread = Thread(target=self.__send_message_periodically, args=[event])
                    message_thread.start()
                print("Monitoring config.json for changes, main thread is blocked.")
                event.wait()

                print("Config file has changed, main thread continues.")
                observer.stop()
                observer.join()
                if self.lark_bot:
                    message_thread.join()
        self.client = openai.OpenAI(
            api_key=self.key_list[self.key_index].get('api_key', None),
            organization=self.key_list[self.key_index].get('organization', None),
            base_url=self.key_list[self.key_index].get('base_url', None)
            )
        
    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def completions_with_backoff(
            self,
            messages,
            engine="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            get_tokens=False,
            **kwargs
        ) -> Union[Tuple[str, int, int], str]:
        """create a completion with gpt. Currently support `davinci`, `turbo` and `gpt-4`

        Args:
            messages (list): messages sent to `turbo`, `gpt4` or a list of prompts sent to `davinci`. (When using davinci, it is recommended to request in batches. @ref: https://platform.openai.com/docs/guides/rate-limits/error-mitigation)
            engine (str, optional): gpt model. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): Defaults to 0.7.
            max_tokens (int, optional): Defaults to 2048.
            top_p (int, optional): Defaults to 1.
            frequency_penalty (int, optional): Defaults to 0.
            presence_penalty (int, optional): Defaults to 0.

        Raises:
            NotImplementedError: _description_

        Returns:
            response(str) for `turbo` and `gpt-4`
            responses(List[str]) for `davinci`
        """
        openai.api_key = self.key_list[self.key_index]['api_key']
        if 'organization' in self.key_list[self.key_index]:
            openai.organization = self.key_list[self.key_index]['organization']
        if 'base_url' in self.key_list[self.key_index]:
            openai.base_url = self.key_list[self.key_index]['base_url']
        sleep_Time = 1
        if get_tokens:
            encoding = tiktoken.encoding_for_model(engine)
            if len(messages) >= 1 and type(messages[0]) == dict:
                input_tokens = len(encoding.encode(''.join([m['content'] for m in messages])))
            elif len(messages) >= 1 and type(messages[0]) == str:
                input_tokens = len(encoding.encode(''.join(messages)))
            else:
                input_tokens = 0

        while True:
            try:
                if any(x in engine for x in ['davinci', 'turbo-instruct']):
                        completion = self.client.completions.create(
                            model=engine,
                            prompt=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            **kwargs
                        )
                        if len(completion.choices) > 1:
                            responses = [choice.text for choice in completion.choices]
                            return responses
                        else:
                            response = completion.choices[0].text
                            if get_tokens:
                                output_tokens = len(encoding.encode(response))
                                return response, input_tokens, output_tokens
                            return response
                elif any(x in engine for x in ['gpt-3.5', 'gpt-4']):
                        completion = self.client.chat.completions.create(
                            model=engine,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            top_p=top_p,
                            frequency_penalty=frequency_penalty,
                            presence_penalty=presence_penalty,
                            **kwargs
                        )
                        response = completion.choices[0].message.content
                        if get_tokens:
                            output_tokens = len(encoding.encode(response))
                            return response, input_tokens, output_tokens
                        return completion.choices[0].message.content
                else:
                    raise NotImplementedError('Currently only support `davinci`, `turbo` and `gpt-4`')
            except openai.RateLimitError as ex:
                if 'Rate limit reached' in str(ex):
                    raise ex
                elif 'exceeded' in str(ex):
                    print(str(ex)+f'\nCurrent api key: {openai.api_key}')
                    self.set_api_key()
                else:
                    print(f'RateLimiteError unhandled...')
                    raise ex
            except openai.BadRequestError as ex:
                if 'have access to' in str(ex):
                    print(ex)
                    self.set_api_key()
                else:
                    raise ex
            except openai.AuthenticationError as ex:
                if 'deactivated' in str(ex):
                    print(f'Api key: {self.key_list[self.key_index]["key"]} has been deactivated. Origin error message: {ex}')
                    self.set_api_key()
                else:
                    raise ex
            except Exception as ex:
                print(ex)
                    # print("##"*5 + ex + "##" *5)
                sleep(sleep_Time)
                sleep_Time *= 2
                if sleep_Time > 1024:
                    print("Sleep time > 1024s")
                    exit(0)

    @staticmethod
    def multi_process_pipeline(config_path: str, processes_num: int, data: Iterable, func: Callable, *args, **kwargs):
        """Execute the function `func` using data and *args, **kwargs

        Args:
            config_path (str): Config file path.
            processes_num (int): The number of processes.
            data (Iterable): The data to process.
            func (Callable): Data processing function.
            args (Any): Additional arguments passed to the function.
            kwargs (Any): Additional keyword arguments passed to the function.

        Returns:
            Any: Return the execution result.
        """
        if os.path.exists(config_path) is False:
            raise FileExistsError(f'Failed to find {config_path}. Please check your file path and try again.')
        elif config_path.endswith('json') is False:
            raise ValueError('Please construct the config file in `JSON` format.')
        
        manager = Manager()
        error_queue = manager.Queue()
        result_queue = [Queue() for _ in range(processes_num)]
        
        lark_hook = kwargs.pop('lark_hook', None)
        
        
        chunk_size = round(len(data)/processes_num)
        processes = []
        results = []
        for i in range(processes_num):
            wrapper = GPTWrapper(config_path=config_path, bias=i, lark_hook=lark_hook)
            data_chunk = data[i*chunk_size:(i+1)*chunk_size]
            process = Process(target=_wrapper_func, args=(i, result_queue[i], error_queue, wrapper, func, data_chunk, *args), kwargs=kwargs)
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
            
        if not error_queue.empty():
            pid, error = error_queue.get()
            raise type(error)(f"An error occurred in child process {pid}: {str(error)}") from error

        for q in result_queue:
            while not q.empty():
                if type(q.get()) == list:
                    results.extend(q.get())
                else:
                    results.append(q.get())
        
        return results
        
    @staticmethod
    def single_round_multi_process(config_path: str, engine: str, processes_num: int, messages: List[Union[Dict, str]], fout: str, **kwargs):
        """Use system prompts and prompts to generate response with multiple processes. If engine is not a chat model, prompt will be formatted
        as `[System Prompt]: {system_prompt}\\n[Prompt]: {prompt}` if system_prompts is not None.

        Args:
            config_path (str): Config file path.
            engine (str): GPT engine.
            processes_num (int): Number of processes to use.
            messages (List[Union[Dict, str]]): List of messages (for chat completions) or prompts (for completions).
            fout (str): Output file. `jsonl` recommended.

        Returns:
            List[JSON]: List of results. 
        """
        
        chunk_size = round(len(messages)/processes_num)
        processes = []
        lark_hook = kwargs.pop('lark_hook', None)
        for i in range(processes_num):
            wrapper = GPTWrapper(config_path=config_path, bias=i, lark_hook=lark_hook)
            messages_subset = messages[i*chunk_size:(i+1)*chunk_size]
            
            process = Process(target=_generate_response, args=(wrapper, engine, messages_subset, f'worker{i}_{fout}'), kwargs=kwargs)
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
        
        results = []
        idx = 0
        fp = open(fout, 'a', encoding='utf-8')
        for i in range(processes_num):
            with open(f'worker{i}_{fout}', 'r', encoding='utf-8') as f:
                data = [json.loads(s) for s in f.readlines()]
                for item in data:
                    item['id'] = idx
                    result = json.dumps(item, ensure_ascii=False)
                    results.append(result)
                    fp.write(result + '\n')
                    idx += 1
        fp.close()
        return results
    
    @staticmethod
    def multi_thread_pipeline(config_path: str, threads_num: int, data: Iterable, func: Callable, *args, **kwargs):
        """Execute the function `func` using data and *args, **kwargs

        Args:
            config_path (str): Config file path.
            threads_num (int): The number of threads.
            data (Iterable): The data to process.
            func (Callable): Data processing function.
            args (Any): Additional arguments passed to the function.
            kwargs (Any): Additional keyword arguments passed to the function.

        Returns:
            Any: Return the execution result.
        """
        if os.path.exists(config_path) is False:
            raise FileExistsError(f'Failed to find {config_path}. Please check your file path and try again.')
        elif config_path.endswith('json') is False:
            raise ValueError('Please construct the config file in `JSON` format.')
        
        error_queue = queue.Queue()
        result_queue = queue.Queue()
        
        chunk_size = round(len(data)/threads_num)
        threads = []
        results = []
        lark_hook = kwargs.pop('lark_hook', None)
        for i in range(threads_num):
            wrapper = GPTWrapper(config_path=config_path, bias=i, lark_hook=lark_hook)
            data_chunk = data[i*chunk_size:(i+1)*chunk_size]
            thread = Thread(target=_wrapper_func, args=(i, result_queue, error_queue, wrapper, func, data_chunk, *args), kwargs=kwargs)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
            
        if not error_queue.empty():
            tid, error = error_queue.get()
            raise type(error)(f"An error occurred in child thread {tid}: {str(error)}") from error

        while not result_queue.empty():
            if type(result_queue.get()) == list:
                results.extend(result_queue.get())
            else:
                results.append(result_queue.get())
        
        return results
    
    @staticmethod
    def single_round_multi_thread(config_path: str, engine: str, threads_num: int, messages: List[Union[Dict, str]], fout: str, **kwargs):
        """

        Args:
            config_path (str): Config file path.
            engine (str): GPT engine.
            threads_num (int): Number of threads to use.
            messages (List[Union[Dict, str]]): List of messages (for chat completions) or prompts (for completions).
            fout (str): Output file. `jsonl` recommended.

        Returns:
            List[JSON]: List of results. 
        """

        chunk_size = round(len(messages)/threads_num)
        processes = []
        lark_hook = kwargs.pop('lark_hook', None)
        for i in range(threads_num):
            wrapper = GPTWrapper(config_path=config_path, bias=i, lark_hook=lark_hook)
            messages_subset = messages[i*chunk_size:(i+1)*chunk_size]
            
            process = Thread(target=_generate_response, args=(wrapper, engine, messages_subset, f'worker{i}_{fout}'), kwargs=kwargs)
            processes.append(process)
            process.start()
        
        for process in processes:
            process.join()
        
        results = []
        idx = 0
        fp = open(fout, 'a', encoding='utf-8')
        for i in range(threads_num):
            with open(f'worker{i}_{fout}', 'r', encoding='utf-8') as f:
                data = [json.loads(s) for s in f.readlines()]
                for item in data:
                    item['id'] = idx
                    result = json.dumps(item, ensure_ascii=False)
                    results.append(result)
                    fp.write(result + '\n')
                    idx += 1
        fp.close()
        return results