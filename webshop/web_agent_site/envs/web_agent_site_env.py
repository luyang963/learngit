import gymnasium as gym
from gymnasium import spaces
import random
import requests
import string
import time
import numpy as np

from bs4 import BeautifulSoup
from bs4.element import Comment
from os.path import join, dirname, abspath
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import ElementNotInteractableException

from web_agent_site.engine.engine import parse_action, END_BUTTON


def tag_visible(element):
    """Helper method to filter visible HTML text"""
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return element.parent.name not in ignore and not isinstance(element, Comment)


class WebAgentSiteEnv(gym.Env):
    """Gymnasium environment for WebShop site in HTML/text mode"""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, observation_mode='html', **kwargs):
        super().__init__()
        self.observation_mode = observation_mode
        self.kwargs = kwargs

        # Selenium driver setup
        service = Service(join(dirname(abspath(__file__)), 'chromedriver'))
        options = Options()
        if not kwargs.get('render', False):
            options.add_argument("--headless")
        self.browser = webdriver.Chrome(service=service, options=options)

        # Session management
        self.text_to_clickable = {}
        self.assigned_session = kwargs.get('session')
        self.session = None

        # Placeholders
        self.instruction_text = ""
        self.reset()

        # Dummy action/observation spaces (recommended to override)
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        """Reset environment and return initial observation and info"""
        super().reset(seed=seed)
        if self.assigned_session is not None:
            self.session = self.assigned_session
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=5))
        init_url = f'http://127.0.0.1:3000/{self.session}'
        self.browser.get(init_url)

        self.instruction_text = self.get_instruction_text()
        obs = self.observation
        info = {}
        return obs, info

    def step(self, action):
        """Take an action and return (obs, reward, terminated, truncated, info)"""
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        action_name, action_arg = parse_action(action)

        # Handle search
        if action_name == 'search' and action_arg:
            try:
                search_bar = self.browser.find_element(By.ID, 'search_input')
                search_bar.clear()
                search_bar.send_keys(action_arg)
                search_bar.submit()
            except Exception:
                pass

        # Handle click
        elif action_name == 'click' and action_arg in self.text_to_clickable:
            try:
                self.text_to_clickable[action_arg].click()
            except ElementNotInteractableException:
                button = self.text_to_clickable[action_arg]
                self.browser.execute_script("arguments[0].click();", button)
            reward = self.get_reward()
            if action_arg == END_BUTTON:
                terminated = True

        # Handle end explicitly
        elif action_name == 'end':
            terminated = True

        else:
            print('Invalid action. No action performed.')

        # Optional pause
        if self.kwargs.get('pause'):
            time.sleep(self.kwargs['pause'])

        obs = self.observation
        return obs, reward, terminated, truncated, info

    @property
    def observation(self):
        """Return HTML or text observation based on mode"""
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html)
        else:
            raise ValueError(f"Unsupported observation mode: {self.observation_mode}")

    @property
    def state(self):
        """Return full state dict"""
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )

    def get_available_actions(self):
        """Return available click/search actions"""
        # Check search bar
        try:
            self.browser.find_element(By.ID, 'search_input')
            has_search_bar = True
        except Exception:
            has_search_bar = False

        # Collect clickable buttons
        buttons = self.browser.find_elements(By.CLASS_NAME, 'btn')
        product_links = self.browser.find_elements(By.CLASS_NAME, 'product-link')
        buying_options = self.browser.find_elements(By.CSS_SELECTOR, "input[type='radio']")

        self.text_to_clickable = {b.text: b for b in buttons + product_links}
        for opt in buying_options:
            val = opt.get_attribute('value')
            self.text_to_clickable[val] = opt

        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )

    def _parse_html(self, html=None):
        """Parse HTML into BeautifulSoup object"""
        if html is None:
            html = self.state['html']
        return BeautifulSoup(html, 'html.parser')

    def convert_html_to_text(self, html):
        """Strip HTML to visible text"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        return ' [SEP] '.join(t.strip() for t in visible_texts if t.strip())

    def get_reward(self):
        """Get reward from HTML page"""
        html_obj = self._parse_html()
        r = html_obj.find(id='reward')
        if r:
            pre_tag = r.find("pre")
            if pre_tag and pre_tag.string:
                return float(pre_tag.string)
        return 0.0

    def get_instruction_text(self):
        """Get instruction text from current page"""
        html_obj = self._parse_html()
        inst = html_obj.find(id='instruction-text')
        if inst and inst.h4:
            return inst.h4.text
        return ""

    def render(self, mode='human'):
        """Optional rendering method"""
        if mode == 'human':
            print(self.observation)
        else:
            raise NotImplementedError("Only human render mode is supported")

    def close(self):
        """Close the browser"""
        self.browser.quit()
        print("Browser closed.")
