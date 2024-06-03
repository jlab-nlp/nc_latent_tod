# Error Analysis

Here we conduct an error analysis comparing our Noisy Channel method with a Greedy ablation.
Both models completed 2 iterations of our EM approach, where the Greedy method uses direct greedy inference without noisy channel re-ranking, for both end-to-end inference and self-labeling.

First, we sample 10 dialogues from the validation set that at least one model for which at least one of the models was unsuccessful:

```python
['pmul0690',
 'mul1854',
 'pmul4691',
 'pmul2154',
 'pmul4729',
 'pmul4881',
 'pmul3923',
 'mul1888',
 'mul1881',
 'sng01664'] 
```
Then, we've annotated errors by identifying the turn that is the cause of the dialogue failure in one or both systems.


## PMUL0690.json

**Noisy Channel**: Successful: we predict that we should offer a restaurant, and then correctly do so. The gold act refers to this as informing, but functions similarly.

**Greedy:** **Policy failure:** We fail to predict the correct next dialogue act, which is to Inform (or Offer) a matching restaurant to the user. This prevents the appropriate placeholder from being included in the response. The response generator ignores the policy's choice (Requesting more info) and hallucinates a restaurant to offer, however this restaurant does not meet the goal requirements.

### Context

User: Perfect. I am also looking for a moderate priced restaurant in the east

System (gold): There are [value_choice] indian restaurants and one [value_food] Do either of these interest you?

User: I am interested in Italian.

| Model         | Acts                                                                                                                              | Response                                                                                |
|---------------|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| Greedy        | [Request(service='restaurant', values=['name', 'address', 'phone', 'postcode'])]                                                  | I have the caffee uno located in the centre. Would you like me to make reservations?    |
| Noisy Channel | [Offer(entity=Restaurant(name='[value_name]'))]                                                                                   | I would recommend [value_name] Would you like me to book a table for you?               |
| Gold          | [Inform(entity=Entity()), Inform(entity=Restaurant(address='Cambridge retail park newmarket road', name='pizza hut fen ditton'))] | the [value_name] is located at [value_address] Would you like me to book a reservation? |



## MUL1854.json

### Noisy Channel
Informed, but not Successful: fails to tell the user the address (though the user never asks)

### Greedy
Neither informed nor Successful: fails to give the user a park because of a **policy failure** believing it should Request a (different) area, and hallucinating that no parks exist as a result.

### Context
User: Not right now, but thanks for asking. I was wondering if there are any parks on the south side of town, though.

| Model         | Acts                                                                                                                         | Response                                                                                                                                  |
|---------------|------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Greedy        | [Request(service='attraction', values=['area'])]                                                                             | There are no parks on the south side of town. Would you like me to look in another area?                                                  |
| Noisy Channel | [Inform(entity=Attraction(name='[value_name]', area='[value_area]'))]                                                        | There is the [value_name] in the [value_area] of town.                                                                                    |
| Gold          | [Offer(entity=Attraction(address='gog magog hills', area='south side', entrancefee='FREE', name='Wandlebury Country Park'))] | On the [value_area] we have [value_name] at Wandlebury ring, [value_address] babraham. The admission is [value_price] are you interested? |

##  PMUL4691.json

**Noisy Channel**: **DST Failure**, misses the area slot.

**Greedy:** **Policy Failure**, again we predict the `Request` act at an inappropriate time. The response generator ignores this, but the address cannot be delexicalized or inserted from an API result. Having seen this entity before in other conversations, the response generator nonetheless predicts the correct address, though this would not generalize to a response for an unseen entity.

### Context

System (gold): on what area do you need the attraction?

User: I'd like it to be in the centre, please.

> The noisy channel model predicts an incorrect dialogue state, missing the area=centre slot for attractions.

Later in the dialogue: 

User: Would you forward the address for that also please?


| Model         | Acts | Response |
|---------------|------|----------|
| Greedy        |  [Request(service='hotel', values=['name', 'address'])]    |     Sure, Christ's College is located at saint andrew's street. Is there anything else I can help you with today?     |
| Gold          |  [Inform(entity=Attraction(address="Saint Andrew's street")), RequestAlternatives()]    |      Address is [value_address] Can I help you find anything else?    |

## PMUL2154.json

**Noisy Channel**: Successful.

**Greedy:** **Policy Failure** Predicts a goodbye act, which is ignored by the response generator. However the response generator then must hallucinate a reference number, which is incorrect.

User: I would like to make reservation for four at The Cambridge Chop House at 16:45 on Tuesday. May I have a reference number when it's booked please?

| Model         | Acts | Response |
|---------------|------|----------|
| Greedy        |   [Goodbye()]   |     Your booking was successful. Your reference number is 12OOOWTR. Is there anything else I can help you with?     |
| Noisy Channel |   [Affirm(entity=Restaurant(ref='[value_ref]'))]   |      Booking was successful. The table will be reserved for 15 minutes. Reference number is [value_reference]    |
| Gold          |    [NotifySuccess(service='entity', entity=Entity(ref='RVGNO6UQ'))] |   I have booked a table and the reference number is [value_reference] .       |


## PMUL4729.json

**Noisy Channel**: Informed, but not Succesful: we fail to provide the postcode because the user never asks for it. The gold system provided this proactively, preventing the user's request.

**Greedy:** we again predict an inappropriate `Request` act, but it is ignored by the response generator.

### Context

User: the cambridge punter is what I would love to go to

| Model         | Acts | Response |
|---------------|------|----------|
| Greedy        |    [Request(service='attraction', values=['address', 'phone'])] |     The Cambridge Punter is a boat attraction in the center of town. Would you like more information? |
| Noisy Channel |   [Inform(entity=Entity(name='[value_name]', address='[value_address]', phone='[value_phone]'))]   |    The [value_name] is located at [value_address] and their phone number is [value_phone]      |
| Gold          |  [Inform(entity=Attraction(address='251a Chesterton Road', area='center of town', postcode='cb41as', type='boating'))]    |   What a great choice! That is a [value_type] attraction that is located in the [value_area] They are located at [value_address] Their postcode is [value_postcode]       |


## PMUL4881.json

**Noisy Channel**: Successful.

**Greedy:** we again predict an inappropriate `Request` act, We already know the area. The response generator does not offer a hotel.

### Context



| Model         | Acts | Response |
|---------------|------|----------|
| Greedy        |    [Request(service='hotel', values=['area'])] |        I have 7 guesthouses that meet that criteria. Do you have a preference on the area of town?   |
| Noisy Channel |    [Offer(entity=Hotel(name='[value_name]', type='[value_type]', area='[value_area]', stars='[value_stars]'))] |     I have the [value_name] which is a [value_stars] star [value_type] in the [value_area] Would you like me to book it for you?     |
| Gold          |   [Inform(entity=Entity()), Offer(entity=Hotel(name='aylesbray lodge guest house'))]   |    The [value_name] is available. Do you need a reservation? |


## PMUL3923.json

**Noisy Channel**: Requests an attraction type from the user, where the system proactively makes a single suggestion and carries the conversation from there.

**Greedy:** Identical to noisy channel model predictions.

### Context

 User: After you get around to that reference number, I also want to go somewhere in the Centre of town.

| Model         | Acts | Response |
|---------------|------|----------|
| Greedy        |   [Request(service='attraction', values=['type'])]    |     What type of attraction are you looking for? |
| Noisy Channel |  [Request(service='attraction', values=['type'])]    |      What type of attraction are you looking for? |
| Gold          |   [Offer(entity=Attraction(name='all saints church'))]   |   what about [value_name]       |

## MUL1888.json

**Noisy Channel & Greedy:** Both models produce the same polict prediction and similar responses, where the gold system response proactively makes a suggestion. Both approaches seem reasonable, where the gold system response may be more helpful if the user finds the suggestion acceptable.

### Context

User: Ok, that's fine. I don't need to book that yet. I do need to find some places to go in the centre of town too.


| Model         | Acts | Response |
|---------------|------|----------|
| Greedy        |  [Request(service='attraction', values=['type'])]    |     There are 44 attractions in the centre of town. What type of attraction are you interested in? |
| Noisy Channel |   [Request(service='attraction', values=['type'])]   |    There are many attractions in the centre of town. Do you have a specific type in mind?      |
| Gold          |   [Inform(entity=Attraction(address='market street', name='holy trinity church'))]  |      You should try visiting the [value_name] they're on [value_address]    |

## MUL1881.json

**Noisy Channel**: Successful.

**Greedy:** Incorrectly predicts that we should `Request` info from the user, including for slots that only the system should know.

### Context

User: Perhaps a college, then. Could you recommend one?


| Model         | Acts | Response |
|---------------|------|----------|
| Greedy        |   [Request(service='attraction', values=['name', 'address', 'phone'])]   |      Sure, there are five colleges on the west side. Churchill college, Clare hall, Clare College, Magdalene College, and Queens' College. Would you like more information on any of these? |
| Noisy Channel |    [Offer(entity=Entity(name='[value_name]', address='[value_address]', entrance_fee='[value_entrance_fee]'))]    |       I would recommend [value_name] on [value_address] It has [value_price] admission. |
| Gold          |   Offer(entity=Attraction(entrancefee='no entrance fee', name='Churchill College'))]   |     [value_name] is lovely this time of year, and they have [value_price] |

## SNG01664
**Noisy Channel**: Successful.

**Greedy:** We request the price range and type, even though we already know both.

### Context

User: I am unsure about the area, right now. I would like the accommodations to be at least 4 star and have a moderate price.
System (gold): And will you need parking at the [value_type]
User: No. I shouldn't need parking.



| Model         | Acts | Response |
|---------------|------|----------|
| Greedy        |  [Request(service='hotel', values=['price_range', 'type'])]    |     Okay, is there anything else I can help you with? |
| Noisy Channel |    [Offer(entity=Hotel(name='[value_name]', area='[value_area]'))] |      I would recommend the [value_name] in the [value_area] Would you like me to book a room for you?  |
| Gold          |    [Inform(entity=Hotel(area='eastern')), Offer(entity=Hotel(name='A and B Guest House'))] |      How do you feel about the [value_name] It covers all your needs and is in the [value_area] area. Would you like a phone number or address?  |

