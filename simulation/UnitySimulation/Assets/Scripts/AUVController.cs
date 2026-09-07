using UnityEngine;

public class AUVController : MonoBehaviour
{
    public float moveSpeed = 5f;

    private Rigidbody rb;
    private Vector3 movement;

    void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    void Update()
    {
        float horizontal = Input.GetAxis("Horizontal");
        float vertical = Input.GetAxis("Vertical");

        float upDown = 0f;

        if (Input.GetKey(KeyCode.E))
        {
            upDown = 1f;
        }
        else if (Input.GetKey(KeyCode.Q))
        {
            upDown = -1f;
        }

        movement = new Vector3(horizontal, upDown, vertical);
    }

    void FixedUpdate()
    {
        rb.MovePosition(rb.position + movement * moveSpeed * Time.fixedDeltaTime);
    }
}