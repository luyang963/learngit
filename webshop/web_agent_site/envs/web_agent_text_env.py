import gymnasium as gym
from gymnasium import spaces
import json
import random
import string
import time
import torch
import numpy as np

from bs4 import BeautifulSoup
from bs4.element import Comment
from collections import defaultdict
from flask import Flask
from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    get_top_n_product_from_keywords,
    map_action_to_html,
    parse_action,
    get_product_per_page,
    ACTION_TO_TEMPLATE,
    END_BUTTON, NEXT_PAGE, PREV_PAGE, BACK_TO_SEARCH,
)
from web_agent_site.engine.goal import get_reward, get_goals
from web_agent_site.utils import DEFAULT_FILE_PATH, FEAT_CONV, FEAT_IDS, random_idx


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return element.parent.name not in ignore and not isinstance(element, Comment)


app = Flask(__name__)


class WebAgentTextEnv(gym.Env):
    """Gymnasium Environment for WebShop Text mode"""

    metadata = {"render_modes": ["human"]}

    def __init__(self, observation_mode='html', file_path=DEFAULT_FILE_PATH, server=None, **kwargs):
        super().__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs
        self.file_path = file_path
        self.base_url = 'http://127.0.0.1:3000'

        # Server and browser
        self.server = SimServer(
            self.base_url,
            self.file_path,
            self.kwargs.get('filter_goals'),
            self.kwargs.get('limit_goals', -1),
            self.kwargs.get('num_products'),
            self.kwargs.get('human_goals'),
            self.kwargs.get('show_attrs', False),
        ) if server is None else server
        self.browser = SimBrowser(self.server)

        # Session info
        self.session = self.kwargs.get('session')
        self.session_prefix = self.kwargs.get('session_prefix')

        # Image features if enabled
        if self.kwargs.get('get_image', 0):
            self.feats = torch.load(FEAT_CONV)
            self.ids = torch.load(FEAT_IDS)
            self.ids = {url: idx for idx, url in enumerate(self.ids)}

        # History
        self.prev_obs = []
        self.prev_actions = []
        self.num_prev_obs = self.kwargs.get('num_prev_obs', 0)
        self.num_prev_actions = self.kwargs.get('num_prev_actions', 0)

        # Placeholder observation and action spaces (可根据需要扩展)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Discrete(1)

        self.reset()

    @property
    def observation(self):
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html, simple=True)
        elif self.observation_mode == 'text_rich':
            return self.convert_html_to_text(html, simple=False)
        elif self.observation_mode == 'url':
            return self.state['url']
        else:
            raise ValueError(f"Observation mode {self.observation_mode} not supported.")

    @property
    def state(self):
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )

    def step(self, action):
        info = {}
        self.get_available_actions()
        action_name, action_arg = parse_action(action)
        if action_arg is not None:
            action_arg = action_arg.lower()

        if action_name == 'search' and action_arg:
            status = self.browser.search(action_arg)
        elif action_name == 'click' and action_arg in self.text_to_clickable and action_arg != 'search':
            status = self.browser.click(action_arg, self.text_to_clickable)
        else:
            status = dict(reward=0, done=False)

        ob = self.observation
        text_list = [ob]

        self.prev_actions.append(action)
        for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
            if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                text_list.append(self.prev_actions[-i])
            if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                text_list.append(self.prev_obs[-i])

        state = ' [SEP] '.join(text_list[::-1])
        self.prev_obs.append(ob)

        # Gymnasium separates done into terminated and truncated
        terminated = status['done']
        truncated = False
        reward = status['reward']

        return state, reward, terminated, truncated, info

    def reset(self, session=None, instruction_text=None, **kwargs):
        session_int = None
        if session is not None:
            self.session = str(session)
            if isinstance(session, int):
                session_int = session
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=10))
        if self.session_prefix:
            self.session = self.session_prefix + self.session

        init_url = f'{self.base_url}/{self.session}'
        self.browser.get(init_url, session_id=self.session, session_int=session_int)

        self.text_to_clickable = None
        self.instruction_text = self.get_instruction_text() if instruction_text is None else instruction_text
        obs = self.observation
        self.prev_obs = [obs]
        self.prev_actions = []

        info = {}
        return obs, info

    def get_available_actions(self):
        html_obj = self._parse_html()
        search_bar = html_obj.find(id='search_input')
        has_search_bar = search_bar is not None

        buttons = html_obj.find_all(class_='btn')
        product_links = html_obj.find_all(class_='product-link')
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {f'{b.get_text()}'.lower(): b for b in buttons + product_links}
        for opt in buying_options:
            self.text_to_clickable[opt.get('value')] = opt

        return dict(has_search_bar=has_search_bar, clickables=list(self.text_to_clickable.keys()))

    def _parse_html(self, html=None):
        if html is None:
            html = self.state['html']
        return BeautifulSoup(html, 'html.parser')

    def convert_html_to_text(self, html, simple=False):
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        if simple:
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        observation = ''
        for t in visible_texts:
            if t == '\n':
                continue
            if t.parent.name == 'button':
                processed_t = f"[button] {t} [button_]"
            elif t.parent.name == 'label':
                processed_t = f"[button] {t} [button_]"
            elif t.parent.get('class') == ["product-link"]:
                processed_t = f"[button] {t} [button_]"
            else:
                processed_t = str(t)
            observation += processed_t + '\n'
        return observation

    def get_instruction_text(self):
        html_obj = self._parse_html(self.browser.page_source)
        return html_obj.find(id='instruction-text').h4.text

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class SimServer:
    """Lightweight simulator of WebShop Flask application for generating HTML observations"""
    def __init__(self, base_url, file_path, filter_goals=None, limit_goals=-1, num_products=None, human_goals=0, show_attrs=False):
        self.base_url = base_url
        self.all_products, self.product_item_dict, self.product_prices, _ = load_products(filepath=file_path, num_products=num_products, human_goals=human_goals)
        self.search_engine = init_search_engine(num_products=num_products)
        self.goals = get_goals(self.all_products, self.product_prices, human_goals)
        self.show_attrs = show_attrs

        random.seed(233)
        random.shuffle(self.goals)

        if filter_goals is not None:
            self.goals = [goal for i, goal in enumerate(self.goals) if filter_goals(i, goal)]

        if limit_goals != -1 and limit_goals < len(self.goals):
            weights = [goal['weight'] for goal in self.goals]
            cum_weights = [0] + np.cumsum(weights).tolist()
            idxs = []
            while len(idxs) < limit_goals:
                idx = random_idx(cum_weights)
                if idx not in idxs:
                    idxs.append(idx)
            self.goals = [self.goals[i] for i in idxs]

        self.weights = [goal['weight'] for goal in self.goals]
        self.cum_weights = [0] + np.cumsum(self.weights).tolist()
        self.user_sessions = dict()
        self.search_time = 0
        self.render_time = 0
        self.sample_time = 0
        self.assigned_instruction_text = None  # hacky


    def receive(self, session_id, current_url, session_int=None, **kwargs):
        """Map action to corresponding page"""
        status = dict(reward=0.0, done=False)

        with app.app_context(), app.test_request_context():
            if session_id not in self.user_sessions:
                idx = session_int if (session_int is not None and isinstance(session_int, int)) else random_idx(self.cum_weights)
                goal = self.goals[idx]
                instruction_text = goal['instruction_text']
                self.user_sessions[session_id] = {'goal': goal, 'done': False}
            else:
                instruction_text = self.user_sessions[session_id]['goal']['instruction_text']
            if self.assigned_instruction_text is not None:
                instruction_text = self.assigned_instruction_text
                self.user_sessions[session_id]['goal']['instruction_text'] = instruction_text

            session = self.user_sessions[session_id]

            if not kwargs:
                kwargs['instruction_text'] = instruction_text
                html, url = self.index(session_id, **kwargs)
                self.user_sessions[session_id].update({'keywords': None, 'page': None, 'asin': None, 'asins': set(), 'options': dict(), 'actions': defaultdict(int)})
            elif 'keywords' in kwargs:
                html, url = self.search_results(session_id, **kwargs)
            elif 'clickable_name' in kwargs:
                clickable_name = kwargs['clickable_name'].lower()
                if clickable_name == END_BUTTON.lower():
                    html, url, reward = self.done(session_id, **kwargs)
                    status['reward'] = reward
                    status['done'] = True
                elif clickable_name == BACK_TO_SEARCH.lower():
                    html, url, status = self.receive(session_id, current_url)
                elif clickable_name == NEXT_PAGE.lower() and self.get_page_name(current_url) == 'search_results':
                    html, url, status = self.receive(session_id, current_url, keywords=session["keywords"], page=session["page"] + 1)
                elif clickable_name == PREV_PAGE.lower() and self.get_page_name(current_url) == 'search_results':
                    html, url, status = self.receive(session_id, current_url, keywords=session["keywords"], page=session["page"] - 1)
                elif clickable_name == PREV_PAGE.lower() and self.get_page_name(current_url) == 'item_sub_page':
                    html, url = self.item_page(session_id, **kwargs)
                elif clickable_name == PREV_PAGE.lower() and self.get_page_name(current_url) == 'item_page':
                    html, url = self.search_results(session_id, keywords=session["keywords"], page=session["page"], **kwargs)
                elif clickable_name in [k.lower() for k in ACTION_TO_TEMPLATE]:
                    html, url = self.item_sub_page(session_id, **kwargs)
                else:
                    html, url = self.item_page(session_id, **kwargs)
            return html, url, status

    def get_page_name(self, url):
        if url is None:
            return None
        page_names = ['search_results', 'item_page', 'item_sub_page', 'done']
        for page_name in page_names:
            if page_name in url:
                return page_name
        return ''

    # 具体 Flask 路由实现略，可复用原有 map_action_to_html 和引擎逻辑


class SimBrowser:
    """Simulated browser"""
    def __init__(self, server):
        self.server = server
        self.current_url = None
        self.page_source = None
        self.session_id = None

    def get(self, url, session_id=None, session_int=None):
        self.session_id = url.split('/')[-1] if session_id is None else session_id
        self.page_source, _, _ = self.server.receive(self.session_id, self.current_url, session_int=session_int)
        self.current_url = url

    def click(self, clickable_name, text_to_clickable):
        self.page_source, self.current_url, status = self.server.receive(self.session_id, current_url=self.current_url, clickable_name=clickable_name, text_to_clickable=text_to_clickable)
        return status

    def search(self, keywords):
        if isinstance(keywords, str):
            keywords = keywords.split(' ')
        self.page_source, self.current_url, status = self.server.receive(self.session_id, current_url=self.current_url, keywords=keywords)
        return status
