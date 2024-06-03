import json
import string
import random

import boto3
from typing import Optional, TypeVar, Generic

T = TypeVar('T')


class SQSQueue(Generic[T]):
    """
    An SQS-based queue for managing a multithread or multiprocess queue of JSON objects.
    """ 

    def __init__(self, queue_name: str, aws_region: str = 'us-west-2') -> None:
        """
        Creates an SQSQueue to interact with a specified SQS queue.

        :param queue_name: the name of the SQS queue
        :param aws_region: the AWS region where the queue is hosted
        """
        super().__init__()
        self.sqs = boto3.resource('sqs', region_name=aws_region)
        self.queue = self.sqs.get_queue_by_name(QueueName=queue_name)

    def enqueue(self, jsonable_obj: T, message_group_id: str = None):
        """
        Add an object onto the end of the queue.
        """
        if message_group_id is None:
            # use a random 8 character identifier if no message group id is provided
            message_group_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        self.queue.send_message(MessageBody=json.dumps(jsonable_obj), MessageGroupId=message_group_id)

    def dequeue(self) -> Optional[T]:
        """
        Get and remove the first item from the queue. Returns None if the queue is empty.
        Automatically handles message deletion after retrieval.
        """
        messages = self.queue.receive_messages(MaxNumberOfMessages=1, WaitTimeSeconds=5)
        if not messages:
            return None

        message = messages[0]
        jsonable_obj = json.loads(message.body)
        message.delete()
        return jsonable_obj

    def __len__(self):
        """
        Get the approximate number of messages in the queue.
        """
        self.queue.load()
        return int(self.queue.attributes.get('ApproximateNumberOfMessages'))


if __name__ == '__main__':
    queue = SQSQueue[str]('todzero-work-queue.fifo')
    queue.enqueue('dff!')
    print(len(queue))
    print(queue.dequeue())
    print(queue.dequeue())
    print(len(queue))
