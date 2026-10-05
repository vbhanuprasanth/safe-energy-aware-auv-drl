using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class PPOUnityBridge : MonoBehaviour
{
    [Header("References")]
    public AUVAgent auvAgent;

    [Header("PPO Server")]
    public string serverHost = "127.0.0.1";
    public int serverPort = 5005;

    private TcpClient client;
    private NetworkStream stream;

    private void Start()
    {
        if (auvAgent == null)
        {
            auvAgent = GetComponent<AUVAgent>();
        }

        try
        {
            client = new TcpClient();
            client.Connect(serverHost, serverPort);

            stream = client.GetStream();

            Debug.Log("PPO BRIDGE: Connected to PPO server.");
        }
        catch (Exception e)
        {
            Debug.LogError(
                "PPO BRIDGE: Could not connect to PPO server.\n" +
                e.Message
            );
        }
    }

    private void FixedUpdate()
    {
        if (stream == null || !stream.CanWrite)
        {
            return;
        }

        if (auvAgent == null)
        {
            return;
        }

        try
        {
            // Get the same 28 observations used by AUVAgent
            float[] observation =
                auvAgent.GetCurrentObservations();

            // Convert observation to JSON
            string request =
                "{\"observation\":[" +
                string.Join(",", observation) +
                "]}\n";

            byte[] requestBytes =
                Encoding.UTF8.GetBytes(request);

            stream.Write(
                requestBytes,
                0,
                requestBytes.Length
            );

            // Receive PPO response
            byte[] buffer = new byte[4096];

            int bytesRead =
                stream.Read(
                    buffer,
                    0,
                    buffer.Length
                );

            if (bytesRead <= 0)
            {
                return;
            }

            string response =
                Encoding.UTF8.GetString(
                    buffer,
                    0,
                    bytesRead
                );

            // Parse response
            PPOResponse ppoResponse =
                JsonUtility.FromJson<PPOResponse>(
                    response.Trim()
                );

            if (ppoResponse == null ||
                ppoResponse.action == null ||
                ppoResponse.action.Length != 3)
            {
                Debug.LogError(
                    "PPO BRIDGE: Invalid PPO action."
                );

                return;
            }

            // Send the 3 PPO actions to the AUV
            auvAgent.ApplyExternalAction(
                ppoResponse.action
            );
        }
        catch (Exception e)
        {
            Debug.LogError(
                "PPO BRIDGE ERROR: " +
                e.Message
            );
        }
    }

    private void OnDestroy()
    {
        if (stream != null)
        {
            stream.Close();
        }

        if (client != null)
        {
            client.Close();
        }
    }

    [Serializable]
    private class PPOResponse
    {
        public float[] action;
    }
}